"""Load captured images and camera and light metadata from dataset.json."""

from __future__ import annotations

import re
from pathlib import Path

import mitsuba as mi
import numpy as np
import orjson

from .common import Dataset, Measurement
from .download import default_root, download as download_material


class Photos(Dataset):
    """Captured views selected by file name, with cameras and lights centred on the material.

    The original implementation incorrectly applied a homography when mapping
    captured images to the material plane. This implementation instead uses
    the calibrated camera geometry.
    """

    # the capture's camera frame to Mitsuba's (its own inverse)
    TO_MITSUBA = np.array(
        [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64
    )

    def __init__(
        self,
        material: str | None = None,
        root: str | None = None,
        path: str | None = None,
        filter: str = "random.*.exr",
        image_scale: float = 0.5,
        sample_scale: float = 70,
        scene_scale: float = 1.0,
        emitter_radius: float = 4.0,
        radiance: float = 16000.0,
        crop: tuple[tuple[float, float], tuple[float, float]] = ((0, 0), (1, 1)),
        material_uv_crop: tuple[tuple[float, float], tuple[float, float]] | None = None,
        download: bool = True,
    ) -> None:
        """Args:
        material: a released material (e.g. "fabric_hex"), loaded from
            root/<material> and downloaded there if missing.
        root: folder holding the released materials (default: $MESOSCALE_DATA,
            or ~/.cache/mesoscale).
        path: a capture folder to load instead, e.g. your own captures; takes
            precedence over material and is never downloaded.
        filter: regex on the image file names (random_*, val_rand_*, sweep_*).
        image_scale: scale of the loaded images (0.5: half resolution).
        sample_scale: half-size of the sample plane, in scene units.
        scene_scale: scale of the camera and emitter positions.
        emitter_radius: radius of the spherical emitter.
        radiance: radiance of the emitter.
        crop: image crop ((u0, v0), (u1, v1)), used without material_uv_crop.
        material_uv_crop: crop on the material ((u0, v0), (u1, v1)); each view's
            image is cropped to the bounding box of its projection.
        download: download a missing material (see datasets.download).
        """
        super().__init__()
        self.emitter_radius = emitter_radius
        self.radiance = radiance
        self.filter = filter
        self.image_scale = image_scale
        self.sample_scale = sample_scale
        self.scene_scale = scene_scale
        self.crop = crop
        self.material_uv_crop = material_uv_crop

        if path:
            self.path = Path(path).expanduser()
            if not (self.path / "dataset.json").exists():
                raise FileNotFoundError(f"no dataset.json in {self.path}")
        elif material:
            root = Path(root).expanduser() if root else default_root()
            self.path = root / material
            if not (self.path / "dataset.json").exists():
                if not download:
                    raise FileNotFoundError(
                        f"{material} is not in {root}; run python -m datasets.download "
                        f"{material} or enable downloading"
                    )
                download_material(material, root)
        else:
            raise ValueError("pass a material or a path")

        meta = orjson.loads((self.path / "dataset.json").read_bytes())
        camera = meta["camera"]
        w, h = camera["resolution"]
        fov = np.degrees(2 * np.arctan(h / (2 * camera["f"])))
        self.geo = {"cam_fov": float(fov), "cam_res": [w, h]}
        # Mitsuba's principal point offset, in units of the film size: pixel
        # centres at (i + 0.5) / size, the camera's at integer pixels
        self.principal_point = (
            ((w - 1) / 2 - camera["cx"]) / w,
            ((h - 1) / 2 - camera["cy"]) / h,
        )

        # the material's centre in the sample holder frame
        offset = meta.get("mask", {}).get("offset_mm", [0.0, 0.0, 0.0])
        self.center = np.array([-offset[0], offset[1], offset[2]], dtype=np.float64)

        pattern = re.compile(filter)
        views = [v for v in meta["views"] if pattern.match(v["file"])]
        self.dataset = sorted(views, key=lambda v: v["file"])

    def __getitem__(self, index: int) -> Measurement:
        view = self.dataset[index]
        cam_to_world = np.asarray(view["camera_to_world"]) @ self.TO_MITSUBA
        cam_to_world[:3, 3] -= self.center
        to_world = mi.ScalarAffineTransform4f(cam_to_world.tolist())
        em_to_world = mi.ScalarAffineTransform4f().translate(
            (np.asarray(view["light_pos"]) - self.center).tolist()
        )
        if self.scene_scale != 1.0:
            matrix = mi.ScalarMatrix4f(to_world.matrix)
            for i in range(3):
                matrix[i, 3] *= self.scene_scale
            to_world = mi.ScalarAffineTransform4f(matrix)
            em_to_world = mi.ScalarAffineTransform4f().translate(
                em_to_world.translation() * self.scene_scale
            )

        image_crop = None
        if self.material_uv_crop is not None:
            film = {
                "type": "hdrfilm",
                "width": int(self.geo["cam_res"][0] * self.image_scale),
                "height": int(self.geo["cam_res"][1] * self.image_scale),
            }
            sensor = mi.load_dict(self.sensor_dict(to_world, film))
            image_crop = self.image_crop(self.material_uv_crop, sensor)

        img, film = self.load_view(view, image_crop)
        emitter = {
            "type": "sphere",
            "radius": self.emitter_radius * self.scene_scale,
            "to_world": em_to_world,
            "emitter": {
                "type": "area",
                "radiance": {"type": "spectrum", "value": self.radiance},
            },
        }
        return Measurement(
            scene_dict={
                "type": "scene",
                "sensor": self.sensor_dict(to_world, {"type": "hdrfilm", **film}),
                "emitter": emitter,
            },
            image=img,
        )

    def image_crop(self, material_uv_crop, sensor: mi.Sensor):
        """Image crop ((u0, v0), (u1, v1)): the bounding box of the projected
        material_uv_crop, clamped to the image."""
        (u0, v0), (u1, v1) = material_uv_crop
        us, vs = [], []
        for c in (
            mi.ScalarPoint2f(u0, v0),
            mi.ScalarPoint2f(u1, v0),
            mi.ScalarPoint2f(u0, v1),
            mi.ScalarPoint2f(u1, v1),
        ):
            # uv in [0, 1]² spans [-sample_scale, sample_scale]² on the plane z = 0
            it = mi.Interaction3f()
            it.p = mi.Point3f(
                (c.x * 2.0 - 1.0) * self.sample_scale * self.scene_scale,
                (c.y * 2.0 - 1.0) * self.sample_scale * self.scene_scale,
                0.0,
            )
            ds, _ = sensor.sample_direction(it, mi.Point2f(0))
            res = sensor.film().crop_size()
            us.append((ds.uv.x / res.x).numpy().item())
            vs.append((ds.uv.y / res.y).numpy().item())
        return (
            (max(0.0, min(us)), max(0.0, min(vs))),
            (min(1.0, max(us)), min(1.0, max(vs))),
        )

    def load_view(self, view: dict, crop=None) -> tuple:
        """Load, resize, and crop a photograph; return the image and film settings.

        Reconstruct the full image before resizing. Crop coordinates are in
        [0, 1]² of the film and default to ``self.crop``."""
        img = mi.Bitmap(str(self.path / view["file"]))
        img = img.convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float16, False)
        x0, y0, cw, ch = view["crop"]
        cam_w, cam_h = self.geo["cam_res"]
        full = np.zeros((cam_h, cam_w, 3), dtype=np.float16)
        full[y0 : y0 + ch, x0 : x0 + cw] = np.asarray(img)
        img = mi.Bitmap(full)

        if crop is None:
            crop = self.crop
        img = img.resample(mi.ScalarVector2u(img.size() * self.image_scale))
        img = mi.TensorXf16(img)
        h, w, _ = img.shape
        crop_y = (int(crop[0][1] * h), int(crop[1][1] * h))
        crop_x = (int(crop[0][0] * w), int(crop[1][0] * w))
        film = {
            "width": w,
            "height": h,
            "crop_offset_x": crop_x[0],
            "crop_offset_y": crop_y[0],
            "crop_width": crop_x[1] - crop_x[0],
            "crop_height": crop_y[1] - crop_y[0],
        }
        return img[crop_y[0] : crop_y[1], crop_x[0] : crop_x[1], :], film

    def sensor_dict(self, to_world, film: dict) -> dict:
        """The view's sensor: Mitsuba's perspective camera."""
        return {
            "type": "perspective",
            "fov_axis": "y",
            "fov": self.geo["cam_fov"],
            "principal_point_offset_x": self.principal_point[0],
            "principal_point_offset_y": self.principal_point[1],
            "film": film,
            "sampler": {"type": "independent", "sample_count": 128},
            "to_world": to_world,
        }

    def training_scene(self, bsdf: mi.BSDF | dict = {"type": "diffuse"}) -> mi.Scene:
        """Build the sample plane, emitter, and sensor used for training."""
        return mi.load_dict(
            {
                "type": "scene",
                "integrator": {"type": "path", "max_depth": 2},
                "sensor": {
                    "type": "perspective",
                    "film": {"type": "hdrfilm", "width": 128, "height": 128},
                    "sampler": {"type": "independent", "sample_count": 128},
                },
                "emitter": {
                    "type": "sphere",
                    "radius": self.emitter_radius * self.scene_scale,
                    "emitter": {
                        "type": "area",
                        "radiance": {"type": "spectrum", "value": self.radiance},
                    },
                    "to_world": mi.ScalarTransform4f().translate(
                        [0, 0, -1000 * self.scene_scale]
                    ),
                },
                "sample": {
                    "type": "rectangle",
                    "to_world": mi.ScalarTransform4f().scale(
                        self.sample_scale * self.scene_scale
                    ),
                    "bsdf": bsdf,
                },
            }
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return (
            f"Photos(path='{self.path}', filter='{self.filter}', "
            f"image_scale={self.image_scale}, sample_scale={self.sample_scale}, "
            f"scene_scale={self.scene_scale}, crop={self.crop}, "
            f"material_uv_crop={self.material_uv_crop}, "
            f"emitter_radius={self.emitter_radius})"
        )
