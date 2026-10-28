"""Renders a view with a PyTorch BSDF: rays, intersections and emitter samples
from Mitsuba, BSDF evaluation and accumulation in PyTorch.
"""

from __future__ import annotations

import drjit as dr
from drjit.auto.ad import Float, UInt32, TensorXf
import mitsuba as mi
import torch


def render_measurement(bsdf, measurement, scene, spp=64, seed=0):
    """Renders the view of a measurement with the PyTorch BSDF ``bsdf``: (image, mask)."""
    from torch_bridge import vec3_to_torch, color3_to_torch

    sensor = measurement.sensor
    film = sensor.film()
    crop_size = film.crop_size()
    width, height = int(crop_size[0]), int(crop_size[1])
    n_pixels = width * height

    radiance_accum = torch.zeros(n_pixels, 3, device="cuda")
    mask_accum = torch.zeros(n_pixels, dtype=torch.bool, device="cuda")

    for s in range(spp):
        with dr.suspend_grad():
            rng = dr.rng(UInt32(seed + s))
            px = dr.arange(UInt32, n_pixels) % width
            py = dr.arange(UInt32, n_pixels) // width
            jitter = rng.random(mi.Point2f, (2, n_pixels))
            sample_pos = mi.Point2f(
                (Float(px) + jitter.x) / Float(width),
                (Float(py) + jitter.y) / Float(height),
            )
            ray, _ = sensor.sample_ray(
                Float(0),
                rng.random(Float, n_pixels),
                sample_pos,
                rng.random(mi.Point2f, (2, n_pixels)),
            )

            si = scene.ray_intersect(ray)
            # Shade only the sample surface.
            valid = si.is_valid() & ~si.shape.is_emitter()

            ds, Le = measurement.emitter.sample_direction(
                si, rng.random(mi.Point2f, (2, n_pixels))
            )
            wo = si.to_local(ds.d)

            shadow_ray = si.spawn_ray(ds.d)
            shadow_ray.maxt = ds.dist * (1 - 1e-3)
            occluded = scene.ray_test(shadow_ray)

            dr.eval(si, wo, ds, Le, valid, occluded)

        uv = si.uv
        si_uv_t = torch.stack([TensorXf(uv.x).torch(), TensorXf(uv.y).torch()], dim=1)
        si_wi_t = vec3_to_torch(si.wi)
        wo_t = vec3_to_torch(wo)
        Le_t = color3_to_torch(Le)
        valid_t = TensorXf(valid).torch().bool()
        occluded_t = TensorXf(occluded).torch().bool()
        pdf_t = TensorXf(ds.pdf).torch()

        with torch.no_grad():
            f_t = bsdf(si_uv_t, si_wi_t, wo_t, it=-1)["f"]

        cos_theta = torch.clamp(wo_t[:, 2:3], min=0)
        radiance = f_t * cos_theta * Le_t
        active = (valid_t & ~occluded_t & (pdf_t > 0)).unsqueeze(1)
        radiance = radiance * active.float()

        radiance_accum += radiance
        mask_accum |= valid_t

    radiance_accum /= spp

    idx = dr.arange(UInt32, n_pixels)
    rad_np = radiance_accum.cpu().numpy()

    img = dr.zeros(TensorXf, (height, width, 3))
    dr.scatter(img.array, Float(rad_np[:, 0]), idx * 3 + 0)
    dr.scatter(img.array, Float(rad_np[:, 1]), idx * 3 + 1)
    dr.scatter(img.array, Float(rad_np[:, 2]), idx * 3 + 2)

    mask = dr.zeros(TensorXf, (height, width, 1))
    dr.scatter(mask.array, Float(mask_accum.float().cpu().numpy()), idx)

    return img, mask


def render_mask(measurement, scene, seed=0):
    """Mask of the pixels that see the sample."""
    sensor = measurement.sensor
    film = sensor.film()
    crop_size = film.crop_size()
    width, height = int(crop_size[0]), int(crop_size[1])
    n_pixels = width * height

    with dr.suspend_grad():
        rng = dr.rng(UInt32(seed))
        px = dr.arange(UInt32, n_pixels) % width
        py = dr.arange(UInt32, n_pixels) // width
        sample_pos = mi.Point2f(
            (Float(px) + 0.5) / Float(width),
            (Float(py) + 0.5) / Float(height),
        )
        ray, _ = sensor.sample_ray(
            Float(0), rng.random(Float, n_pixels), sample_pos, mi.Point2f(0.5, 0.5)
        )
        si = scene.ray_intersect(ray)
        valid = dr.select(si.is_valid() & ~si.shape.is_emitter(), Float(1), Float(0))

    mask = dr.zeros(TensorXf, (height, width, 1))
    dr.scatter(mask.array, valid, dr.arange(UInt32, n_pixels))
    return mask
