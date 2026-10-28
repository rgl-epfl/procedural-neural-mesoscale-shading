"""Sample material points and image blocks from captured views."""

from __future__ import annotations

import torch
from datasets.torch_measurements import (
    TorchMeasurements,
    build_local_frame,
    to_local,
)


def _rand(n: int, d: int, generator: torch.Generator, device: torch.device):
    return torch.rand(n, d, generator=generator, device=device)


def _rand_1d(n: int, generator: torch.Generator, device: torch.device):
    return torch.rand(n, generator=generator, device=device)


def _plane_sample(
    uv: torch.Tensor,
    half_extent: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand (u,v) ∈ [0,1]² to (p, n, s, t) for the z=0 plane of ±half_extent."""
    p = torch.stack(
        [
            (uv[:, 0] * 2.0 - 1.0) * half_extent,
            (uv[:, 1] * 2.0 - 1.0) * half_extent,
            torch.zeros_like(uv[:, 0]),
        ],
        dim=1,
    )
    n = torch.zeros_like(p)
    n[:, 2] = 1.0
    s, t = build_local_frame(n)
    return p, n, s, t


def _image_uv(
    measurements: TorchMeasurements, m_idx: torch.Tensor, uv_pix: torch.Tensor
):
    """Pixel positions [M, 2] on each view's film to texture coordinates in [0, 1]²."""
    W = measurements.img_width[m_idx].to(uv_pix.dtype)
    H = measurements.img_height[m_idx].to(uv_pix.dtype)
    return torch.stack([uv_pix[:, 0] / W, uv_pix[:, 1] / H], dim=1)


class NativeTorchSurfaceSampler:
    """Sample n_uv material points, each observed from n_dir random views."""

    def __init__(
        self,
        n_uv: int = 1,
        n_dir: int = 1,
        material_uv_crop: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        self.n_uv = n_uv
        self.n_dir = n_dir
        self.material_uv_crop = material_uv_crop

    def sample(
        self,
        scene,
        measurements: TorchMeasurements,
        generator: torch.Generator,
        **kwargs,
    ) -> dict:
        device = measurements.device
        with torch.no_grad():
            return self._sample_impl(measurements, generator, device, kwargs)

    def _sample_impl(self, measurements, generator, device, kwargs):
        uv = _rand(self.n_uv, 2, generator, device)
        if self.material_uv_crop is not None:
            (u0, v0), (u1, v1) = self.material_uv_crop
            uv = torch.stack(
                [
                    u0 + (u1 - u0) * uv[:, 0],
                    v0 + (v1 - v0) * uv[:, 1],
                ],
                dim=1,
            )

        uv_rep = uv.unsqueeze(1).expand(-1, self.n_dir, -1).reshape(-1, 2).contiguous()
        n_tot = uv_rep.shape[0]

        return self._finalize_uv(measurements, uv_rep, generator, device, n_tot)

    def _finalize_uv(self, measurements, uv_rep, generator, device, n_tot):
        """Views, observed radiance and directions for the given surface uv."""
        p, n, s, t = _plane_sample(uv_rep, measurements.plane_half_extent)

        r = _rand_1d(n_tot, generator, device)
        m_idx = (r * len(measurements)).to(torch.int64)
        s1 = _rand(n_tot, 2, generator, device)
        s2 = _rand(n_tot, 2, generator, device)
        sensor = measurements.sensor_sample_direction(m_idx, p, s1)
        emitter = measurements.emitter_sample_direction(m_idx, p, s2)

        uv_tex = _image_uv(measurements, m_idx, sensor["uv"])
        L_ref = measurements.eval(m_idx, uv_tex)

        wi_local = to_local(sensor["d"], s, t, n)
        wo_local = to_local(emitter["d"], s, t, n)
        valid = sensor["pdf"] > 0.0

        return {
            "L_ref": L_ref,
            "Le": emitter["w"],
            "si_uv": uv_rep,
            "si_wi": wi_local,
            "wo": wo_local,
            "valid": valid,
            "shape": [3, self.n_uv, self.n_dir],
        }

    def __repr__(self):
        return f"NativeTorchSurfaceSampler(n_uv={self.n_uv}, n_dir={self.n_dir})"


class NativeTorchBlockSensorSampler:
    """Sample image blocks around material points projected into random views."""

    def __init__(
        self,
        n_blocks: int,
        block_size: tuple[int, int],
        n_uv: int,
        material_uv_crop: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        self.n_blocks = n_blocks
        self.block_size = tuple(block_size)
        self.n_uv = int(n_uv)
        self.material_uv_crop = material_uv_crop

    def sample(
        self,
        scene,
        measurements: TorchMeasurements,
        generator: torch.Generator,
        **kwargs,
    ) -> dict:
        device = measurements.device
        with torch.no_grad():
            n_pixels = self.block_size[0] * self.block_size[1]

            m_idx_block, block_offset, n_total_blocks, valid_block = (
                self._uv_guided_blocks(measurements, generator, device)
            )

            m_idx = m_idx_block.repeat_interleave(n_pixels)
            sample_index = torch.arange(n_pixels, device=device).repeat(n_total_blocks)

            block_offset_per_pixel = block_offset.repeat_interleave(n_pixels, dim=0)

            crop_w = measurements.img_width[m_idx].to(torch.float32)
            crop_h = measurements.img_height[m_idx].to(torch.float32)
            crop_size = torch.stack([crop_w, crop_h], dim=1)

            y = sample_index // self.block_size[0]
            x = sample_index - y * self.block_size[0]
            pixel_xy = torch.stack([x.to(torch.float32), y.to(torch.float32)], dim=1)

            jitter = _rand(n_total_blocks * n_pixels, 2, generator, device) - 0.5
            sample_point = (pixel_xy + jitter + block_offset_per_pixel) / crop_size

            si, emitter, L_ref = measurements.sample_ray(m_idx, sample_point, generator)

            Le = emitter["w"]
            wo = to_local(emitter["d"], si["s"], si["t"], si["n"])

            valid = si["valid"] & valid_block.repeat_interleave(n_pixels)

            return {
                "L_ref": L_ref,
                "Le": Le,
                "si_uv": si["uv"],
                "si_wi": si["wi"],
                "wo": wo,
                "valid": valid,
                "shape": [3, n_total_blocks, self.block_size[1], self.block_size[0]],
            }

    def _uv_guided_blocks(
        self,
        measurements: TorchMeasurements,
        generator: torch.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """For each of n_uv points of the material, n_blocks random views and the block
        centred on the point's pixel in each: (view index, top-left pixel, number of
        blocks, valid). Blocks whose point projects outside the image are kept, with
        valid False, so the count stays fixed.
        """
        n_meas = len(measurements)
        n_total_blocks = self.n_uv * self.n_blocks
        bs = torch.tensor(self.block_size, dtype=torch.float32, device=device)

        uv = _rand(self.n_uv, 2, generator, device)
        if self.material_uv_crop is not None:
            (u0, v0), (u1, v1) = self.material_uv_crop
            uv = torch.stack(
                [u0 + (u1 - u0) * uv[:, 0], v0 + (v1 - v0) * uv[:, 1]], dim=1
            )
        extent = measurements.plane_half_extent
        p_world = torch.stack(
            [
                (uv[:, 0] * 2.0 - 1.0) * extent,
                (uv[:, 1] * 2.0 - 1.0) * extent,
                torch.zeros_like(uv[:, 0]),
            ],
            dim=1,
        )

        r = _rand_1d(n_total_blocks, generator, device)
        m_idx_block = (r * n_meas).to(torch.int64)
        # row i * n_blocks + j: anchor i, view j
        p_world_per_block = p_world.repeat_interleave(self.n_blocks, dim=0)

        s_dummy = _rand(n_total_blocks, 2, generator, device)
        sensor = measurements.sensor_sample_direction(
            m_idx_block, p_world_per_block, s_dummy
        )
        pixel_center = sensor["uv"]  # [n_total, 2] in pixel coords
        valid_block = sensor["pdf"] > 0.0  # whether the projection landed in the image

        crop_w = measurements.img_width[m_idx_block].to(torch.float32)
        crop_h = measurements.img_height[m_idx_block].to(torch.float32)
        crop_size = torch.stack([crop_w, crop_h], dim=1)
        block_offset = pixel_center - bs / 2.0
        block_offset = torch.clamp(block_offset, min=0.0)
        block_offset = torch.minimum(block_offset, crop_size - bs)

        return m_idx_block, block_offset, n_total_blocks, valid_block

    def __repr__(self):
        return (
            f"NativeTorchBlockSensorSampler(n_blocks={self.n_blocks}, "
            f"block_size={self.block_size}, n_uv={self.n_uv})"
        )
