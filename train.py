"""Train one material stage using configs/train.yaml.

Save checkpoint.pkl in the run directory; runs/<name>/latest points to that run."""

import os
import pickle
from pathlib import Path

import hydra
import mitsuba as mi
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def train(cfg: DictConfig, out_dir: str):
    """Train the configured stage and save its checkpoint."""
    # Initialise the GPU backend only when running.
    mi.set_variant("cuda_ad_rgb")
    seed_all(cfg.seed)
    method = instantiate(cfg.stage.method)
    if cfg.stage.name == "detail":
        if cfg.coarse is None:
            raise ValueError(
                "the detail stage needs coarse=<coarse run>/checkpoint.pkl"
            )
        with open(cfg.coarse, "rb") as f:
            method.load(pickle.load(f))
        # the detail stage counts its own iterations
        method.global_it = 0
    print(f"Training {cfg.name}:")
    method.fit()

    os.makedirs(out_dir, exist_ok=True)
    chkpt = {
        **method.save(),
        "bsdf_cfg": OmegaConf.to_container(cfg.stage.method.bsdf, resolve=True),
    }
    with open(Path(out_dir) / "checkpoint.pkl", "wb") as f:
        pickle.dump(chkpt, f)
    return method


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    # runs/<name>/latest -> this run
    tmp = out_dir.parent / f".latest-{os.getpid()}"
    tmp.symlink_to(out_dir.name)
    tmp.replace(out_dir.parent / "latest")
    train(cfg, str(out_dir))


if __name__ == "__main__":
    main()
