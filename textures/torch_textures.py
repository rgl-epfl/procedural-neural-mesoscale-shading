"""The learnable latent texture."""

from __future__ import annotations

import torch
import torch.nn as torchnn


def bilinear_sample_2d(texture: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup of texture [1, C, H, W] at uv [N, 2] in [0, 1]²: [N, C].
    The same as F.grid_sample(..., mode="bilinear", padding_mode="border",
    align_corners=False).
    """
    _, _, H, W = texture.shape
    # Match grid_sample with align_corners=False.
    x = uv[:, 0] * W - 0.5
    y = uv[:, 1] * H - 0.5

    x0 = torch.floor(x).to(torch.long)
    y0 = torch.floor(y).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0.to(x.dtype)
    wy = y - y0.to(y.dtype)

    x0c = x0.clamp(0, W - 1)
    x1c = x1.clamp(0, W - 1)
    y0c = y0.clamp(0, H - 1)
    y1c = y1.clamp(0, H - 1)

    t = texture[0]  # [C, H, W]
    c00 = t[:, y0c, x0c]  # [C, N]
    c01 = t[:, y1c, x0c]
    c10 = t[:, y0c, x1c]
    c11 = t[:, y1c, x1c]

    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy

    out = c00 * w00 + c01 * w01 + c10 * w10 + c11 * w11  # [C, N]
    return out.t().contiguous()  # [N, C]


class TorchFiltered2D(torchnn.Module):
    """Learnable latent texture with bilinear interpolation."""

    def __init__(self, resolution: int = 512, n_channels: int = 1, init_scale: float = 1e-4):
        super().__init__()
        self.resolution = resolution
        self.n_channels = n_channels
        self.n_outputs = n_channels

        self.level = torchnn.Parameter(
            torch.randn(1, n_channels, resolution, resolution) * init_scale
        )

    def prepare(self, it):
        pass

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        """Latent values [n, n_channels] at uv [n, 2] in [0, 1]²."""
        return bilinear_sample_2d(self.level, uv)
