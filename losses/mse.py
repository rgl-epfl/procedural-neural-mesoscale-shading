"""MSE loss for coarse training."""

from __future__ import annotations

import torch
import torch.nn as torchnn


class MSELoss(torchnn.Module):
    """Mean squared error of log radiance, over samples positive in both."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, L_res: torch.Tensor, L_ref: torch.Tensor) -> torch.Tensor:
        active = (L_res > self.eps).all(dim=1) & (L_ref > self.eps).all(dim=1)
        log_res = torch.log(L_res.clamp(min=self.eps))
        log_ref = torch.log(L_ref.clamp(min=self.eps))
        diff = (log_res - log_ref) ** 2
        diff[~active] = 0
        return diff.sum() / (diff.shape[0] * diff.shape[1])
