"""Training loops for the coarse and detail stages."""

from __future__ import annotations

import re

import drjit as dr
import torch
import tqdm

from dataloaders import ViewDataLoader
from datasets import Dataset
from datasets.torch_measurements import TorchMeasurements


def _dataset_plane_extent(dataset) -> float:
    """Half-extent of the training scene's sample plane: sample_scale * scene_scale."""
    return float(dataset.sample_scale * dataset.scene_scale)


class _Method:
    """Shared training data, optimiser, and training loop."""

    def __init__(
        self,
        dataset: Dataset,
        sampler,
        bsdf: torch.nn.Module,
        loss,
        n: int,
        lr: float,
        seed: int = 0,
    ):
        self.n = n
        self.dataloader = ViewDataLoader(dataset)
        self.sampler = sampler
        self.rng = torch.Generator(device="cuda")
        self.rng.manual_seed(int(seed))
        self.bsdf = bsdf.cuda()
        self.loss = loss

        self.torch_measurements = TorchMeasurements.from_drjit(
            self.dataloader.load(), plane_half_extent=_dataset_plane_extent(dataset)
        )
        dr.flush_malloc_cache()

        # only used to intersect rays with the sample plane
        self.scene = dataset.training_scene({"type": "diffuse"})

        self.optimizer = torch.optim.AdamW(self.bsdf.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler("cuda")
        self.global_it = 0

    def save(self):
        return {
            "bsdf_state": self.bsdf.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_it": self.global_it,
        }

    def fit(self):
        """Run the configured number of training iterations."""
        self.bsdf.compile()
        iterator = tqdm.tqdm(range(self.n), desc="Training")
        for it in iterator:
            result = self.training_step(it)
            if it % 100 == 0:
                iterator.set_postfix(loss=f"{result['loss']:.4f}")
            self.global_it += 1


class TorchCoarseMethod(_Method):
    """Train the coarse model with the configured MSE loss."""

    def load(self, chkpt):
        if "bsdf_state" in chkpt:
            self.bsdf.load_state_dict(chkpt["bsdf_state"])
        if "optimizer_state" in chkpt:
            self.optimizer.load_state_dict(chkpt["optimizer_state"])
        self.global_it = chkpt.get("global_it", 0)

    def training_step(self, it):
        """One optimisation step on a batch of samples."""
        self.bsdf.prepare(it)

        s = self.sampler.sample(self.scene, self.torch_measurements, self.rng)

        self.optimizer.zero_grad()
        res = self.bsdf(s["si_uv"], s["si_wi"], s["wo"], it=it)

        L_res = res["f"] * s["Le"] if s["Le"].numel() > 1 else res["f"]
        loss = self.loss(L_res, s["L_ref"])

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}


class TorchDetailMethod(_Method):
    """Train detail with a style loss across exposures, freezing selected parameters."""

    def __init__(self, *args, exposures=(0.0,), freeze=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.exposures = list(exposures)
        self._frozen_keys = list(freeze)

    def fit(self):
        self._freeze_params()
        super().fit()

    def _freeze_params(self):
        """Freezes the parameters whose names match the ``freeze`` patterns."""
        for pattern in self._frozen_keys:
            for name, param in self.bsdf.named_parameters():
                if re.match(pattern, name):
                    param.requires_grad_(False)

    def load(self, chkpt):
        if "bsdf_state" in chkpt:
            self.bsdf.load_state_dict(chkpt["bsdf_state"], strict=False)
        if "optimizer_state" in chkpt:
            try:
                self.optimizer.load_state_dict(chkpt["optimizer_state"])
            except ValueError:
                pass  # Checkpoints from another stage may have incompatible optimiser state.
        self.global_it = chkpt.get("global_it", 0)

    def training_step(self, it):
        self.bsdf.prepare(it)

        self.optimizer.zero_grad()

        for exposure in self.exposures:
            s = self.sampler.sample(self.scene, self.torch_measurements, self.rng)

            res_torch = self.bsdf(s["si_uv"], s["si_wi"], s["wo"], it=it)
            loss_spatial = self.loss(
                res_torch,
                s["L_ref"] * 2**exposure,
                s["Le"] * 2**exposure,
                s["shape"],
            )
            self.scaler.scale(loss_spatial).backward()

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"loss": loss_spatial.item(), "lr": self.optimizer.param_groups[0]["lr"]}
