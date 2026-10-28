"""Captured views and camera and emitter sampling in PyTorch."""

from __future__ import annotations

import math

import numpy as np
import torch

PLANE_HALF_EXTENT = 100.0


def plane_intersect(
    ray_o: torch.Tensor,  # [N, 3]
    ray_d: torch.Tensor,  # [N, 3]
    half_extent: float = PLANE_HALF_EXTENT,
    min_t: float = 1e-4,
    max_t: float = 1e10,
):
    """Rays against the sample plane z = 0, |x|, |y| <= half_extent: (p, n, uv, valid), uv in [0, 1]²."""
    denom = ray_d[:, 2]
    # no division by zero; those rows are masked
    safe_denom = torch.where(denom.abs() > 1e-30, denom, torch.ones_like(denom))
    t = -ray_o[:, 2] / safe_denom
    p = ray_o + t.unsqueeze(1) * ray_d
    valid = (
        (t > min_t)
        & (t < max_t)
        & (p[:, 0].abs() <= half_extent)
        & (p[:, 1].abs() <= half_extent)
    )
    uv = torch.stack(
        [(p[:, 0] / half_extent + 1.0) * 0.5, (p[:, 1] / half_extent + 1.0) * 0.5],
        dim=1,
    )
    n = torch.zeros_like(ray_o)
    n[:, 2] = 1.0
    return p, n, uv, valid


def build_local_frame(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tangents (s, t) for normals n [N, 3], with the basis of Mitsuba's Frame3f."""
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    sign = torch.where(nz >= 0.0, torch.ones_like(nz), -torch.ones_like(nz))
    a = -1.0 / (sign + nz + 1e-30)
    b = nx * ny * a
    s = torch.stack([1.0 + sign * nx * nx * a, sign * b, -sign * nx], dim=1)
    t = torch.stack([b, sign + ny * ny * a, -ny], dim=1)
    return s, t


def to_local(v: torch.Tensor, s: torch.Tensor, t: torch.Tensor, n: torch.Tensor):
    """Express world-space vector v in a local frame (s, t, n)."""
    return torch.stack(
        [(v * s).sum(1), (v * t).sum(1), (v * n).sum(1)],
        dim=1,
    )


def to_world(v_local: torch.Tensor, s: torch.Tensor, t: torch.Tensor, n: torch.Tensor):
    """Express local-space vector in world coords."""
    return v_local[:, 0:1] * s + v_local[:, 1:2] * t + v_local[:, 2:3] * n


class TorchMeasurements:
    """Packed captured views stored as CUDA tensors."""

    def __init__(
        self,
        pixels: torch.Tensor,  # flat float16 atlas [total_pixels * 3]
        pixel_offset: torch.Tensor,  # [N] int32
        img_width: torch.Tensor,  # [N] int32
        img_height: torch.Tensor,  # [N] int32
        sensor_to_world: torch.Tensor,  # [N, 4, 4]
        sensor_to_world_inv: torch.Tensor,  # [N, 4, 4]
        sensor_persp_proj: torch.Tensor,  # [N, 4, 4]
        sensor_sample_to_camera: torch.Tensor,  # [N, 4, 4]
        sensor_resolution_x: torch.Tensor,  # [N]
        sensor_resolution_y: torch.Tensor,  # [N]
        sensor_scaled_pp_offset_x: torch.Tensor,  # [N]
        sensor_scaled_pp_offset_y: torch.Tensor,  # [N]
        sensor_near_clip: torch.Tensor,  # [N]
        sensor_far_clip: torch.Tensor,  # [N]
        sensor_normalization: torch.Tensor,  # [N]
        emitter_center: torch.Tensor,  # [N, 3]
        emitter_radius: torch.Tensor,  # [N]
        emitter_radiance: torch.Tensor,  # [N, 3]
        plane_half_extent: float = PLANE_HALF_EXTENT,
    ) -> None:
        self.pixels = pixels
        self.pixel_offset = pixel_offset
        self.img_width = img_width
        self.img_height = img_height
        self.sensor_to_world = sensor_to_world
        self.sensor_to_world_inv = sensor_to_world_inv
        self.sensor_persp_proj = sensor_persp_proj
        self.sensor_sample_to_camera = sensor_sample_to_camera
        self.sensor_resolution_x = sensor_resolution_x
        self.sensor_resolution_y = sensor_resolution_y
        self.sensor_scaled_pp_offset_x = sensor_scaled_pp_offset_x
        self.sensor_scaled_pp_offset_y = sensor_scaled_pp_offset_y
        self.sensor_near_clip = sensor_near_clip
        self.sensor_far_clip = sensor_far_clip
        self.sensor_normalization = sensor_normalization
        self.emitter_center = emitter_center
        self.emitter_radius = emitter_radius
        self.emitter_radiance = emitter_radiance
        self.plane_half_extent = plane_half_extent

    @property
    def device(self):
        return self.pixels.device

    def __len__(self):
        return int(self.img_width.shape[0])

    @classmethod
    def from_drjit(
        cls,
        dr_measurements,
        plane_half_extent: float = PLANE_HALF_EXTENT,
    ) -> "TorchMeasurements":
        """Convert packed measurements to PyTorch with the given sample-plane half-extent."""
        from drjit.auto.ad import TensorXf

        def _t(x, dtype=torch.float32):
            tx = TensorXf(x).torch().contiguous().to(dtype=dtype)
            return tx

        def _mat(m, dim):
            # DrJit matrices are (dim, dim, N); torch.bmm wants (N, dim, dim)
            arr = np.asarray(m.numpy()).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[..., None]
            if arr.shape[:2] == (dim, dim):
                arr = np.transpose(arr, (2, 0, 1))
            return torch.from_numpy(arr).contiguous().cuda()

        pixels = dr_measurements.pixels.to(torch.float16).cuda().contiguous()

        def _int(x):
            # int64 offsets (float32 would lose precision on large atlases)
            if isinstance(x, torch.Tensor):
                return x.to(torch.int64).cuda().contiguous()
            arr = np.asarray(x.numpy())
            return torch.from_numpy(arr.astype(np.int64)).cuda().contiguous()

        return cls(
            pixels=pixels,
            pixel_offset=_int(dr_measurements.pixel_offset),
            img_width=_int(dr_measurements.img_width),
            img_height=_int(dr_measurements.img_height),
            sensor_to_world=_mat(dr_measurements.sensor_to_world, 4),
            sensor_to_world_inv=_mat(dr_measurements.sensor_to_world_inv, 4),
            sensor_persp_proj=_mat(dr_measurements.sensor_persp_proj, 4),
            sensor_sample_to_camera=_mat(dr_measurements.sensor_sample_to_camera, 4),
            sensor_resolution_x=_t(dr_measurements.sensor_resolution_x),
            sensor_resolution_y=_t(dr_measurements.sensor_resolution_y),
            sensor_scaled_pp_offset_x=_t(dr_measurements.sensor_scaled_pp_offset_x),
            sensor_scaled_pp_offset_y=_t(dr_measurements.sensor_scaled_pp_offset_y),
            sensor_near_clip=_t(dr_measurements.sensor_near_clip),
            sensor_far_clip=_t(dr_measurements.sensor_far_clip),
            sensor_normalization=_t(dr_measurements.sensor_normalization),
            emitter_center=torch.stack(
                [
                    _t(dr_measurements.emitter_center_x),
                    _t(dr_measurements.emitter_center_y),
                    _t(dr_measurements.emitter_center_z),
                ],
                dim=1,
            ),
            emitter_radius=_t(dr_measurements.emitter_radius),
            emitter_radiance=torch.stack(
                [
                    _t(dr_measurements.emitter_radiance_r),
                    _t(dr_measurements.emitter_radiance_g),
                    _t(dr_measurements.emitter_radiance_b),
                ],
                dim=1,
            ),
            plane_half_extent=plane_half_extent,
        )

    def eval(self, idx: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Bilinear lookup of views ``idx`` [M] at ``uv`` [M, 2] in [0, 1]²: [M, 3] RGB."""
        off = self.pixel_offset[idx].to(torch.int64)
        W = self.img_width[idx]
        H = self.img_height[idx]
        Wf = W.to(uv.dtype)
        Hf = H.to(uv.dtype)

        x = uv[:, 0] * Wf - 0.5
        y = uv[:, 1] * Hf - 0.5
        xf = torch.floor(x)
        yf = torch.floor(y)
        fx = x - xf
        fy = y - yf

        x0 = torch.clamp(xf.to(torch.int64), torch.zeros_like(W), W - 1)
        y0 = torch.clamp(yf.to(torch.int64), torch.zeros_like(W), H - 1)
        x1 = torch.clamp(xf.to(torch.int64) + 1, torch.zeros_like(W), W - 1)
        y1 = torch.clamp(yf.to(torch.int64) + 1, torch.zeros_like(W), H - 1)

        def _rgb(yi, xi):
            base = off + (yi * W + xi) * 3
            r = self.pixels[base]
            g = self.pixels[base + 1]
            b = self.pixels[base + 2]
            return torch.stack([r, g, b], dim=1).to(uv.dtype)

        c00 = _rgb(y0, x0)
        c10 = _rgb(y0, x1)
        c01 = _rgb(y1, x0)
        c11 = _rgb(y1, x1)

        omfx = 1.0 - fx
        omfy = 1.0 - fy
        w00 = (omfx * omfy).unsqueeze(1)
        w10 = (fx * omfy).unsqueeze(1)
        w01 = (omfx * fy).unsqueeze(1)
        w11 = (fx * fy).unsqueeze(1)
        return c00 * w00 + c10 * w10 + c01 * w01 + c11 * w11

    def sensor_sample_direction(
        self,
        idx: torch.Tensor,  # [M]
        it_p: torch.Tensor,  # [M, 3] surface point
        sample2: torch.Tensor,  # Unused; retained for the sensor sampling interface.
    ):
        """Projects surface points ``it_p`` [M, 3] into the views ``idx``: the direction
        to the camera and the image position (``p, d, uv, pdf, n, dist`` and the
        importance ``w``), as Mitsuba's perspective sensor.
        """
        tw_inv = self.sensor_to_world_inv[idx]  # [M, 4, 4]
        pp = self.sensor_persp_proj[idx]
        tw = self.sensor_to_world[idx]
        rx = self.sensor_resolution_x[idx]
        ry = self.sensor_resolution_y[idx]
        sx = self.sensor_scaled_pp_offset_x[idx]
        sy = self.sensor_scaled_pp_offset_y[idx]
        nc = self.sensor_near_clip[idx]
        fc = self.sensor_far_clip[idx]
        nm = self.sensor_normalization[idx]

        ones = torch.ones_like(it_p[:, :1])
        p4 = torch.cat([it_p, ones], dim=1).unsqueeze(-1)  # [M, 4, 1]
        rp4 = torch.bmm(tw_inv, p4).squeeze(-1)  # [M, 4]
        rp = rp4[:, :3]

        active = (rp[:, 2] >= nc) & (rp[:, 2] <= fc)

        svx = (
            pp[:, 0, 0] * rp[:, 0]
            + pp[:, 0, 1] * rp[:, 1]
            + pp[:, 0, 2] * rp[:, 2]
            + pp[:, 0, 3]
        )
        svy = (
            pp[:, 1, 0] * rp[:, 0]
            + pp[:, 1, 1] * rp[:, 1]
            + pp[:, 1, 2] * rp[:, 2]
            + pp[:, 1, 3]
        )
        svw = (
            pp[:, 3, 0] * rp[:, 0]
            + pp[:, 3, 1] * rp[:, 1]
            + pp[:, 3, 2] * rp[:, 2]
            + pp[:, 3, 3]
        )
        iw = 1.0 / svw

        uv_x = svx * iw - sx
        uv_y = svy * iw - sy
        active = active & (uv_x >= 0) & (uv_x <= 1) & (uv_y >= 0) & (uv_y <= 1)
        uv = torch.stack([uv_x * rx, uv_y * ry], dim=1)

        ld = rp
        dist = torch.linalg.norm(ld, dim=1)
        idist = 1.0 / dist
        ld_n = ld * idist.unsqueeze(1)

        cam_pos = tw[:, :3, 3]  # [M, 3] — 4th column
        d = (cam_pos - it_p) * idist.unsqueeze(1)
        cam_n = tw[:, :3, 2]
        pdf = torch.where(active, torch.ones_like(dist), torch.zeros_like(dist))

        ct = ld_n[:, 2]
        ict = 1.0 / ct
        imp = torch.where(ct > 0, nm * ict * ict * ict, torch.zeros_like(ct))
        w = (imp * idist * idist).unsqueeze(1)  # scalar spectrum → broadcast to [M,1]

        return {
            "p": cam_pos,
            "d": d,
            "uv": uv,
            "pdf": pdf,
            "n": cam_n,
            "dist": dist,
            "w": w,
        }

    def sensor_sample_ray(
        self,
        idx: torch.Tensor,
        sample_pos: torch.Tensor,  # [M, 2] in [0, 1]
    ):
        tw = self.sensor_to_world[idx]
        stc = self.sensor_sample_to_camera[idx]
        sx = self.sensor_scaled_pp_offset_x[idx]
        sy = self.sensor_scaled_pp_offset_y[idx]
        nc = self.sensor_near_clip[idx]

        px = sample_pos[:, 0] + sx
        py = sample_pos[:, 1] + sy

        nx = stc[:, 0, 0] * px + stc[:, 0, 1] * py + stc[:, 0, 3]
        ny = stc[:, 1, 0] * px + stc[:, 1, 1] * py + stc[:, 1, 3]
        nz = stc[:, 2, 0] * px + stc[:, 2, 1] * py + stc[:, 2, 3]
        d_cam = torch.stack([nx, ny, nz], dim=1)
        d_cam = d_cam / torch.linalg.norm(d_cam, dim=1, keepdim=True)

        cam_pos = tw[:, :3, 3]
        Rtw = tw[:, :3, :3]
        d_world = torch.bmm(Rtw, d_cam.unsqueeze(-1)).squeeze(-1)
        o = cam_pos + d_world * (nc.unsqueeze(1) * (1.0 / d_cam[:, 2:3]))
        return o, d_world

    def emitter_sample_direction(
        self,
        idx: torch.Tensor,  # [M]
        it_p: torch.Tensor,  # [M, 3]
        sample2: torch.Tensor,  # [M, 2]
    ):
        cx = self.emitter_center[idx]  # [M, 3]
        radius = self.emitter_radius[idx]  # [M]
        Le = self.emitter_radiance[idx]  # [M, 3]

        dc = cx - it_p
        d2 = (dc * dc).sum(1)
        inv_dc = 1.0 / torch.sqrt(d2)
        sin_theta_max = radius * inv_dc
        sin_theta_max_2 = sin_theta_max * sin_theta_max
        inv_sin_theta_max = 1.0 / sin_theta_max
        cos_theta_max = torch.sqrt(torch.clamp(1.0 - sin_theta_max_2, min=0.0))

        # Taylor expansion for small solid angles, as in Mitsuba
        big = sin_theta_max_2 > 0.00068523
        one_minus_fma = torch.square((cos_theta_max - 1.0) * sample2[:, 0] + 1.0)
        sin_theta_2 = torch.where(
            big, 1.0 - one_minus_fma, sin_theta_max_2 * sample2[:, 0]
        )
        cos_theta = torch.sqrt(torch.clamp(1.0 - sin_theta_2, min=0.0))

        cos_alpha = sin_theta_2 * inv_sin_theta_max + cos_theta * torch.sqrt(
            torch.clamp(
                1.0 - sin_theta_2 * inv_sin_theta_max * inv_sin_theta_max, min=0.0
            )
        )
        sin_alpha = torch.sqrt(torch.clamp(1.0 - cos_alpha * cos_alpha, min=0.0))

        two_pi = 2.0 * math.pi
        sin_phi = torch.sin(sample2[:, 1] * two_pi)
        cos_phi = torch.cos(sample2[:, 1] * two_pi)

        n = -dc * inv_dc.unsqueeze(1)
        s, t = build_local_frame(n)
        v_local = torch.stack(
            [
                cos_phi * sin_alpha,
                sin_phi * sin_alpha,
                cos_alpha,
            ],
            dim=1,
        )
        d_normal = to_world(v_local, s, t, n)

        light_pt = d_normal * radius.unsqueeze(1) + cx
        d = light_pt - it_p
        dist2 = (d * d).sum(1)
        dist = torch.sqrt(dist2)
        d = d / dist.unsqueeze(1)

        active = (cos_theta_max < 1.0) & (dist > 0.0)
        pdf = torch.where(
            active,
            1.0 / (two_pi * (1.0 - cos_theta_max)),
            torch.zeros_like(dist),
        )
        safe_pdf = torch.where(pdf > 0, pdf, torch.ones_like(pdf))
        w = torch.where(
            (pdf > 0).unsqueeze(1),
            Le / safe_pdf.unsqueeze(1),
            torch.zeros_like(Le),
        )
        return {"p": light_pt, "d": d, "n": d_normal, "pdf": pdf, "dist": dist, "w": w}

    def sample_ray(
        self,
        measurement_idx: torch.Tensor,
        sample_pos: torch.Tensor,  # [M, 2] in [0, 1]
        generator: torch.Generator,
    ):
        """Rays through the views' pixels, intersected with the sample plane: the hit
        (``p, n, s, t, uv, valid, wi``), the emitter sample and the observed radiance.
        """
        ray_o, ray_d = self.sensor_sample_ray(measurement_idx, sample_pos)
        p, n, uv, valid = plane_intersect(ray_o, ray_d, self.plane_half_extent)
        s, t = build_local_frame(n)
        # wi points toward the camera in the local surface frame.
        wi_local = to_local(-ray_d, s, t, n)

        m = int(measurement_idx.shape[0])
        s_em = torch.rand(m, 2, generator=generator, device=measurement_idx.device)
        emitter = self.emitter_sample_direction(measurement_idx, p, s_em)

        I_ref = self.eval(measurement_idx, sample_pos)

        si = {
            "p": p,
            "n": n,
            "s": s,
            "t": t,
            "uv": uv,
            "valid": valid,
            "wi": wi_local,
        }
        return si, emitter, I_ref
