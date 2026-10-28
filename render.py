"""Render captured views with a trained PyTorch or Mitsuba BSDF."""

from __future__ import annotations

import numpy as np
import torch

from hydra.utils import instantiate
from torch_render import render_measurement, render_mask

CROPS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "100": ((0.0, 0.0), (1.0, 1.0)),
    "25": ((0.375, 0.375), (0.625, 0.625)),
    "20": ((0.4, 0.4), (0.6, 0.6)),
}
DISPLAY_IMAGE_SCALE: float = 1.0


def _drop_keys(cfg, keys):
    """cfg without the given keys at any depth (older checkpoints store options
    that have since been removed)."""
    if isinstance(cfg, dict):
        return {k: _drop_keys(v, keys) for k, v in cfg.items() if k not in keys}
    if isinstance(cfg, list):
        return [_drop_keys(v, keys) for v in cfg]
    return cfg


def build_bsdf(chkpt: dict) -> torch.nn.Module:
    """Build a BSDF from a checkpoint for evaluation."""
    bsdf = instantiate(_drop_keys(chkpt["bsdf_cfg"], {"exponential"}))
    if torch.cuda.is_available():
        bsdf = bsdf.cuda()
    bsdf.load_state_dict(chkpt["bsdf_state"], strict=False)
    bsdf.eval()
    bsdf.prepare(chkpt.get("global_it", 0))
    return bsdf


def build_display_sweep_cfg(data, material_uv_crop) -> dict:
    """Configure full-resolution sweep views with a material crop, for the
    captures selected by the data config (dataset, material, root, path, download)."""
    return {
        "_target_": data.dataset,
        "filter": "sweep.*.exr",
        "material_uv_crop": [
            list(material_uv_crop[0]),
            list(material_uv_crop[1]),
        ],
        "image_scale": DISPLAY_IMAGE_SCALE,
        "scene_scale": 1.0,
        "sample_scale": 70,
        "emitter_radius": 4.0,
        "radiance": 16000.0,
        "material": data.material,
        "root": data.root,
        "path": data.path,
        "download": data.download,
    }


def render_view(bsdf, m, scene, spp, seed=0):
    """The view of measurement ``m`` rendered with the PyTorch BSDF: (image,
    mask of the pixels that see the plate, captured image), numpy [h, w, c]."""
    rendered, _ = render_measurement(bsdf, m, scene, spp=spp, seed=seed)
    mask = render_mask(m, scene, seed=seed)
    ref = m.image
    h, w, _ = ref.shape
    rgb = np.asarray(rendered[:h, :w, :].numpy()).astype(np.float32)
    mk = np.asarray(mask[:h, :w, :].numpy()).astype(np.float32)
    ref_np = np.asarray(ref.numpy()).astype(np.float32)
    return rgb, mk, ref_np


def render_mitsuba(
    chkpt_path: str, m, sample_scale: float = 70.0, spp: int = 64, seed: int = 0
):
    """Render a measured view with the Mitsuba BSDF under direct lighting.

    Return an RGB TensorXf of shape [h, w, 3]."""
    import mitsuba as mi

    import mitsuba_bsdf  # noqa: F401  # registers the plugin

    sensor = dict(m.scene_dict["sensor"])
    sensor["film"] = {**sensor["film"], "rfilter": {"type": "box"}}
    sensor["sampler"] = {"type": "independent", "sample_count": spp}
    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": {"type": "direct", "emitter_samples": 1, "bsdf_samples": 0},
            "sensor": sensor,
            "emitter": m.scene_dict["emitter"],
            "plate": {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().scale(sample_scale),
                "bsdf": {"type": "mesoscale", "chkpt": str(chkpt_path)},
            },
        }
    )
    return mi.render(scene, seed=seed)
