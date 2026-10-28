"""Captured views and packed training measurements."""

from __future__ import annotations

from typing import Any
import mitsuba as mi
from dataclasses import dataclass
import numpy as np
import torch
import drjit as dr
from drjit.auto.ad import (
    Float,
    UInt32,
)


def _to_torch_image(image) -> torch.Tensor:
    """Convert an image to a contiguous CPU float16 tensor."""
    if isinstance(image, torch.Tensor):
        t = image
    elif hasattr(image, "numpy"):
        t = torch.from_numpy(np.asarray(image.numpy(), dtype=np.float16))
    else:
        t = torch.from_numpy(np.asarray(image, dtype=np.float16))
    if t.dtype != torch.float16:
        t = t.to(torch.float16)
    return t.cpu().contiguous()


@dataclass
class Measurement:
    """A captured image and its sensor and emitter scene description."""

    scene_dict: dict
    # the image as a CPU float16 tensor (H, W, 3)
    image: torch.Tensor

    scene: mi.Scene

    DRJIT_STRUCT = {
        "scene": mi.Scene,
    }

    def __init__(
        self,
        scene_dict: dict,
        image,
    ) -> None:
        self.scene_dict = scene_dict
        self.image = _to_torch_image(image)

        self.scene: mi.Scene = mi.load_dict(self.scene_dict)

    def save(self):
        def sanitize(params):
            if isinstance(
                params,
                mi.Transform4f | mi.Transform3f | mi.ScalarAffineTransform4f,
            ):
                return params.matrix
            elif isinstance(params, dict):
                return {k: sanitize(v) for k, v in params.items()}
            else:
                return params

        scene_dict = sanitize(self.scene_dict)

        return {
            "scene_dict": scene_dict,
            "image": self.image,
        }

    @staticmethod
    def load(d: dict):
        scene_dict = d["scene_dict"]
        image = d["image"]
        return Measurement(scene_dict, image)

    @property
    def emitter(self):
        return self.scene.emitters()[0]

    @property
    def sensor(self):
        return self.scene.sensors()[0]


class Measurements:
    """Packed images and per-view sensor and emitter parameters for training."""

    pixels: torch.Tensor  # flat float16, length = sum(h*w*3)
    pixel_offset: torch.Tensor  # int64 cumulative offsets into pixels
    img_width: UInt32
    img_height: UInt32
    sensor_to_world_inv: "mi.Matrix4f"
    sensor_persp_proj: "mi.Matrix4f"
    sensor_to_world: "mi.Matrix4f"
    sensor_sample_to_camera: "mi.Matrix4f"
    sensor_resolution_x: Float
    sensor_resolution_y: Float
    sensor_scaled_pp_offset_x: Float
    sensor_scaled_pp_offset_y: Float
    sensor_near_clip: Float
    sensor_far_clip: Float
    sensor_normalization: Float
    emitter_center_x: Float
    emitter_center_y: Float
    emitter_center_z: Float
    emitter_radius: Float
    emitter_radiance_r: Float
    emitter_radiance_g: Float
    emitter_radiance_b: Float

    DRJIT_STRUCT = {
        "img_width": UInt32,
        "img_height": UInt32,
        "sensor_to_world_inv": Any,
        "sensor_persp_proj": Any,
        "sensor_to_world": Any,
        "sensor_sample_to_camera": Any,
        "sensor_resolution_x": Float,
        "sensor_resolution_y": Float,
        "sensor_scaled_pp_offset_x": Float,
        "sensor_scaled_pp_offset_y": Float,
        "sensor_near_clip": Float,
        "sensor_far_clip": Float,
        "sensor_normalization": Float,
        "emitter_center_x": Float,
        "emitter_center_y": Float,
        "emitter_center_z": Float,
        "emitter_radius": Float,
        "emitter_radiance_r": Float,
        "emitter_radiance_g": Float,
        "emitter_radiance_b": Float,
    }

    @classmethod
    def from_raw(cls, scene_dicts: list[dict], images: list) -> "Measurements":
        """Pack captured images and scene parameters for training."""
        obj = cls.__new__(cls)

        def _s(v):
            return float(v.numpy().flat[0]) if hasattr(v, "numpy") else float(v)

        offsets, ws, hs = [], [], []
        running = 0
        for img in images:
            a = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
            h, w = a.shape[0], a.shape[1]
            offsets.append(running)
            ws.append(w)
            hs.append(h)
            running += h * w * 3

        # Assemble the image atlas on the CPU before uploading.
        atlas_np = np.zeros(running, dtype=np.float16)
        for i, img in enumerate(images):
            a = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
            flat = a.astype(np.float16).reshape(-1)
            atlas_np[offsets[i] : offsets[i] + len(flat)] = flat
        obj.pixels = torch.from_numpy(atlas_np).cuda().contiguous()
        del atlas_np

        # Large image atlases require 64-bit offsets.
        obj.pixel_offset = torch.tensor(offsets, dtype=torch.int64, device="cuda")
        obj.img_width = UInt32(ws)
        obj.img_height = UInt32(hs)

        tw_inv, pp, stc_d, tw_d = [], [], [], []
        rx, ry, sx, sy, nc_l, fc_l, nm = [], [], [], [], [], [], []
        cx, cy, cz, rd, ra_r, ra_g, ra_b = [], [], [], [], [], [], []

        def _np_mat(v, dim):
            """A dim × dim numpy matrix from a scene_dict value."""
            if hasattr(v, "matrix"):
                v = v.matrix
            if hasattr(v, "numpy"):
                a = np.squeeze(v.numpy())
            else:
                a = np.squeeze(np.asarray(v, dtype=np.float32))
            return a.astype(np.float32).reshape(dim, dim)

        for sd in scene_dicts:
            s = sd["sensor"]
            f = s["film"]
            film_w, film_h = f["width"], f["height"]
            crop_w = f.get("crop_width", film_w)
            crop_h = f.get("crop_height", film_h)
            crop_ox = f.get("crop_offset_x", 0)
            crop_oy = f.get("crop_offset_y", 0)

            tw = _np_mat(s["to_world"], 4)

            fov = s.get("fov", 39.6)
            fov_axis = s.get("fov_axis", "x").lower()
            aspect = film_w / film_h
            if fov_axis == "smaller":
                fov_axis = "y" if aspect > 1 else "x"
            elif fov_axis == "larger":
                fov_axis = "x" if aspect > 1 else "y"
            if fov_axis == "y":
                import math

                fov = math.degrees(
                    2.0 * math.atan(math.tan(math.radians(fov) / 2.0) * aspect)
                )
            elif fov_axis == "diagonal":
                import math

                diag = 2.0 * math.tan(math.radians(fov) / 2.0)
                w = diag / math.sqrt(1.0 + 1.0 / (aspect * aspect))
                fov = math.degrees(2.0 * math.atan(w / 2.0))
            near_clip = s.get("near_clip", 1e-2)
            far_clip = s.get("far_clip", 1e4)

            stc_mat = mi.perspective_projection(
                mi.ScalarVector2i(film_w, film_h),
                mi.ScalarVector2i(crop_w, crop_h),
                mi.ScalarVector2i(crop_ox, crop_oy),
                fov,
                near_clip,
                far_clip,
            ).inverse()
            stc_np = _np_mat(stc_mat.matrix, 4)
            pp_np = np.linalg.inv(stc_np)

            # Mitsuba's inverse keeps the last row exact (numpy's does not)
            tw_inv.append(_np_mat(mi.Transform4f(tw).inverse(), 4))
            pp.append(pp_np)
            stc_d.append(stc_np)
            tw_d.append(tw)
            rx.append(float(crop_w))
            ry.append(float(crop_h))
            # Mitsuba's principal point offset, scaled to the crop as its
            # perspective sensor scales it
            sx.append(float(s.get("principal_point_offset_x", 0.0)) * film_w / crop_w)
            sy.append(float(s.get("principal_point_offset_y", 0.0)) * film_h / crop_h)
            nc_l.append(float(near_clip))
            fc_l.append(float(far_clip))

            p0 = stc_np @ np.array([0, 0, 0, 1], dtype=np.float32)
            p1 = stc_np @ np.array([1, 1, 0, 1], dtype=np.float32)
            rect_min = np.array([p0[0], p0[1] / p0[2]])
            rect_max = np.array([p1[0], p1[1] / p1[2]])
            volume = (rect_max[0] - rect_min[0]) * (rect_max[1] - rect_min[1])
            nm.append(1.0 / volume)

            e = sd.get("emitter", {})
            etw = mi.Transform4f(e.get("to_world", mi.Matrix4f()))
            et = etw.translation()
            cx.append(_s(et.x))
            cy.append(_s(et.y))
            cz.append(_s(et.z))
            rd.append(float(e.get("radius", 1.0)))
            # the emitter's RGB radiance (its scalar spectrum converts under D65)
            em_scene = mi.load_dict({"type": "scene", "emitter": sd["emitter"]})
            em = em_scene.shapes()[0].emitter()
            dummy_si = dr.zeros(mi.SurfaceInteraction3f, 1)
            dummy_si.wi = mi.Vector3f(0, 0, 1)
            Le = em.eval(dummy_si)
            ra_r.append(_s(Le.x))
            ra_g.append(_s(Le.y))
            ra_b.append(_s(Le.z))
            del em_scene, em

        def _stack(data, dim):
            a = np.array(data).reshape(len(data), dim, dim)
            t = np.transpose(a, (1, 2, 0))
            return mi.Matrix4f(t)

        obj.sensor_to_world_inv = _stack(tw_inv, 4)
        obj.sensor_persp_proj = _stack(pp, 4)
        obj.sensor_sample_to_camera = _stack(stc_d, 4)
        obj.sensor_to_world = _stack(tw_d, 4)
        obj.sensor_resolution_x = Float(rx)
        obj.sensor_resolution_y = Float(ry)
        obj.sensor_scaled_pp_offset_x = Float(sx)
        obj.sensor_scaled_pp_offset_y = Float(sy)
        obj.sensor_near_clip = Float(nc_l)
        obj.sensor_far_clip = Float(fc_l)
        obj.sensor_normalization = Float(nm)
        obj.emitter_center_x = Float(cx)
        obj.emitter_center_y = Float(cy)
        obj.emitter_center_z = Float(cz)
        obj.emitter_radius = Float(rd)
        obj.emitter_radiance_r = Float(ra_r)
        obj.emitter_radiance_g = Float(ra_g)
        obj.emitter_radiance_b = Float(ra_b)
        return obj

    def __len__(self):
        return int(self.pixel_offset.shape[0])


class Dataset:
    """Base dataset with constructor settings used to identify cached views."""

    _config: tuple

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        orig_init = cls.__init__

        def new_init(self, *args, **kwargs):
            self._config = (args, kwargs)

            orig_init(self, *args, **kwargs)

        cls.__init__ = new_init

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Measurement:
        """The view at ``index`` as a Measurement."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @property
    def config(self):
        return self._config
