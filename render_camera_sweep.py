"""Render full-frame camera sweeps from a trained checkpoint.

Save linear EXR and exposure-adjusted sRGB JPEG frames in the run directory.
See configs/render_camera_sweep.yaml for settings."""

import os
from pathlib import Path

import hydra
import mitsuba as mi
import numpy as np
import tqdm
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image


def write_frame(path: Path, rgb, exposure: float):
    mi.util.write_bitmap(str(path.with_suffix(".exr")), rgb)
    x = np.clip(np.asarray(rgb, dtype=np.float32) * exposure, 0.0, 1.0)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    Image.fromarray((x * 255 + 0.5).astype(np.uint8)).save(path.with_suffix(".jpg"), quality=95)


@hydra.main(config_path="configs", config_name="render_camera_sweep", version_base=None)
def main(cfg: DictConfig) -> None:
    # Initialise the GPU backend only when running.
    mi.set_variant("cuda_ad_rgb")
    import mitsuba_bsdf  # noqa: F401  # registers the plugin

    out = Path(HydraConfig.get().runtime.output_dir)
    tmp = out.parent / f".latest-{os.getpid()}"
    tmp.symlink_to(out.name)
    tmp.replace(out.parent / "latest")

    dataset = instantiate(
        {
            "_target_": cfg.data.dataset,
            "filter": r"sweep_.*\.exr",
            "image_scale": cfg.image_scale,
            "emitter_radius": 4.0,
            "radiance": 16000.0,
            "material": cfg.data.material,
            "root": cfg.data.root,
            "path": cfg.data.path,
            "download": cfg.data.download,
        }
    )
    scene = mi.load_dict(
        {
            **dataset[0].scene_dict,
            "integrator": {"type": "path", "max_depth": 2},
            "sample": {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().scale(cfg.plate_scale),
                "bsdf": {"type": "mesoscale", "chkpt": str(cfg.chkpt)},
            },
        }
    )
    params = mi.traverse(scene)

    last = len(dataset) if cfg.last is None else min(cfg.last, len(dataset))
    for i in tqdm.tqdm(range(cfg.first, last), desc="Camera sweep"):
        params.update(mi.traverse(mi.load_dict(dataset[i].scene_dict)))
        rgb = mi.render(scene, spp=cfg.spp, seed=i)
        write_frame(out / f"{i:03d}", rgb, cfg.exposure)


if __name__ == "__main__":
    main()
