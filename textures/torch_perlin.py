"""4D Perlin noise over position and half vector, for the detail stage."""

from __future__ import annotations

import torch
import torch.nn as torchnn


def _fade(t):
    """Perlin's fade curve 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _hash4(ix, iy, iz, iw):
    """Hash of 4 integer lattice coordinates."""
    h = ix * 1619 + iy * 31337 + iz * 6971 + iw * 1013
    h = h ^ (h >> 8)
    h = h * 0x21F0AAAD
    h = h ^ (h >> 15)
    h = h * 0x735A2D97
    h = h ^ (h >> 15)
    return h


def _perlin_noise_4d(x, y, z, w, grad4_x, grad4_y, grad4_z, grad4_w):
    gx = torch.floor(x)
    gy = torch.floor(y)
    gz = torch.floor(z)
    gw = torch.floor(w)
    fx = x - gx
    fy = y - gy
    fz = z - gz
    fw = w - gw

    u = _fade(fx)
    v = _fade(fy)
    s = _fade(fz)
    t = _fade(fw)

    ix = gx.int()
    iy = gy.int()
    iz = gz.int()
    iw = gw.int()

    def grad_dot(gi, dx, dy, dz, dw):
        return grad4_x[gi] * dx + grad4_y[gi] * dy + grad4_z[gi] * dz + grad4_w[gi] * dw

    def corner(d0, d1, d2, d3):
        gi = _hash4(ix + d0, iy + d1, iz + d2, iw + d3) & 15
        return grad_dot(gi, fx - d0, fy - d1, fz - d2, fw - d3)

    n0000 = corner(0, 0, 0, 0)
    n1000 = corner(1, 0, 0, 0)
    n0100 = corner(0, 1, 0, 0)
    n1100 = corner(1, 1, 0, 0)
    n0010 = corner(0, 0, 1, 0)
    n1010 = corner(1, 0, 1, 0)
    n0110 = corner(0, 1, 1, 0)
    n1110 = corner(1, 1, 1, 0)
    n0001 = corner(0, 0, 0, 1)
    n1001 = corner(1, 0, 0, 1)
    n0101 = corner(0, 1, 0, 1)
    n1101 = corner(1, 1, 0, 1)
    n0011 = corner(0, 0, 1, 1)
    n1011 = corner(1, 0, 1, 1)
    n0111 = corner(0, 1, 1, 1)
    n1111 = corner(1, 1, 1, 1)

    nx000 = torch.lerp(n0000, n1000, u)
    nx100 = torch.lerp(n0100, n1100, u)
    nx010 = torch.lerp(n0010, n1010, u)
    nx110 = torch.lerp(n0110, n1110, u)
    nx001 = torch.lerp(n0001, n1001, u)
    nx101 = torch.lerp(n0101, n1101, u)
    nx011 = torch.lerp(n0011, n1011, u)
    nx111 = torch.lerp(n0111, n1111, u)

    nxy00 = torch.lerp(nx000, nx100, v)
    nxy10 = torch.lerp(nx010, nx110, v)
    nxy01 = torch.lerp(nx001, nx101, v)
    nxy11 = torch.lerp(nx011, nx111, v)

    nxyz0 = torch.lerp(nxy00, nxy10, s)
    nxyz1 = torch.lerp(nxy01, nxy11, s)

    return torch.lerp(nxyz0, nxyz1, t)


class TorchPerlinLayers4D(torchnn.Module):
    """n_levels octaves of n_channels Perlin noise in (u, v, h.x, h.y); the
    directional frequency is dir_freq_ratio times the spatial one. No learnable
    parameters.
    """

    def __init__(
        self,
        resolution=512,
        n_channels=8,
        n_levels=8,
        const=False,
        dir_freq_ratio=0.25,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_levels = n_levels
        self.const = const
        self.dir_freq_ratio = dir_freq_ratio

        self.scale = 1.0
        self.base_freq = resolution / (2.0 ** (n_levels - 1))
        self.persistence = 1.0

        self.register_buffer(
            "grad4_x",
            torch.tensor(
                [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "grad4_y",
            torch.tensor(
                [1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "grad4_z",
            torch.tensor(
                [1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "grad4_w",
            torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1],
                dtype=torch.float32,
            ),
        )

        if self.const:
            self.n_levels += 1

    @property
    def n_outputs(self):
        return self.n_channels * self.n_levels

    def prepare(self, it):
        pass

    def forward(self, uv, h):
        """Noise [N, n_outputs] at uv [N, 2] and half vector h [N, 3]."""
        h = torch.nn.functional.normalize(h, dim=1)
        results = []
        for level_i in range(self.n_levels):
            results.extend(self._eval_level(uv, h, level_i))
        return torch.stack(results, dim=1)

    def _eval_level(self, uv, h, level_i):
        if self.const:
            if level_i == 0:
                return [torch.ones(uv.shape[0], device=uv.device)] * self.n_channels
            level_i -= 1

        freq = self.base_freq * (2.0**level_i)
        amplitude = self.persistence**level_i
        dir_freq = freq * self.dir_freq_ratio

        result = []
        for ch in range(self.n_channels):
            x = uv[:, 0] * self.scale * freq + ch * 43758.5453
            y = uv[:, 1] * self.scale * freq + ch * 12345.6789
            z = h[:, 0] * dir_freq + ch * 78932.1234
            w = h[:, 1] * dir_freq + ch * 56789.4321

            noise_val = (
                _perlin_noise_4d(
                    x, y, z, w, self.grad4_x, self.grad4_y, self.grad4_z, self.grad4_w
                )
                * amplitude
            )
            result.append(noise_val)
        return result
