"""Render sweep views alongside captured references.

Save EXR and PNG frames in the run directory under render/ and reference/.
See configs/render_sweep.yaml for settings."""

import os
import pickle
from pathlib import Path

import hydra
import mitsuba as mi
import numpy as np
import tqdm
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

mi.set_variant("cuda_ad_rgb")

from render import (
    CROPS,
    build_bsdf,
    build_display_sweep_cfg,
    render_mitsuba,
    render_view,
)  # noqa: E402


def _write(path: Path, rgb, exposure: float):
    mi.util.write_bitmap(str(path.with_suffix(".exr")), rgb)
    png = np.clip(np.asarray(rgb, dtype=np.float32) * exposure, 0.0, 1.0)
    mi.util.write_bitmap(str(path.with_suffix(".png")), png)


@hydra.main(config_path="configs", config_name="render_sweep", version_base=None)
def main(cfg: DictConfig) -> None:
    out = Path(HydraConfig.get().runtime.output_dir)
    tmp = out.parent / f".latest-{os.getpid()}"
    tmp.symlink_to(out.name)
    tmp.replace(out.parent / "latest")

    dataset = instantiate(
        build_display_sweep_cfg(cfg.data, CROPS[str(cfg.crop)])
    )
    if cfg.renderer == "torch":
        with open(cfg.chkpt, "rb") as f:
            bsdf = build_bsdf(pickle.load(f))
        scene = dataset.training_scene({"type": "diffuse"})
    elif cfg.renderer != "mitsuba":
        raise ValueError(f"renderer={cfg.renderer!r}: mitsuba or torch")

    for d in ("render", "reference"):
        (out / d).mkdir(exist_ok=True)
    last = len(dataset) if cfg.last is None else min(cfg.last, len(dataset))
    for i in tqdm.tqdm(range(cfg.first, last), desc="Sweep"):
        m = dataset[i]
        h, w = m.image.shape[:2]
        if cfg.renderer == "torch":
            rgb = render_view(bsdf, m, scene, spp=cfg.spp, seed=i)[0]
        else:
            rgb = render_mitsuba(cfg.chkpt, m, spp=cfg.spp, seed=i)[:h, :w]
        _write(out / "render" / f"{i:04d}", rgb, cfg.exposure)
        _write(out / "reference" / f"{i:04d}", m.image, cfg.exposure)


if __name__ == "__main__":
    main()
