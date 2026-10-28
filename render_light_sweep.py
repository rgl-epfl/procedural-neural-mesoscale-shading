"""Render a trained checkpoint with a light rotating around the sample normal.

Save linear EXR and exposure-adjusted sRGB JPEG frames in the run directory.
See configs/render_light_sweep.yaml for settings."""

import os
from pathlib import Path

import hydra
import mitsuba as mi
import numpy as np
import tqdm
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from render_camera_sweep import write_frame


@hydra.main(config_path="configs", config_name="render_light_sweep", version_base=None)
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
    m = dataset[cfg.view]
    scene = mi.load_dict(
        {
            **m.scene_dict,
            "integrator": {"type": "path", "max_depth": 2},
            "sample": {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().scale(cfg.plate_scale),
                "bsdf": {"type": "mesoscale", "chkpt": str(cfg.chkpt)},
            },
        }
    )
    params = mi.traverse(scene)
    light = mi.Transform4f(params["emitter.to_world"])
    normal = np.array([0.0, 0.0, 1.0])  # Sample-plane normal.

    for i in tqdm.tqdm(range(cfg.frames), desc="Light sweep"):
        t = i / max(cfg.frames - 1, 1)
        params["emitter.to_world"] = (
            mi.Transform4f().rotate(axis=normal, angle=t * cfg.angle) @ light
        )
        params.update()
        rgb = mi.render(scene, spp=cfg.spp, seed=i)
        write_frame(out / f"{i:03d}", rgb, cfg.exposure)


if __name__ == "__main__":
    main()
