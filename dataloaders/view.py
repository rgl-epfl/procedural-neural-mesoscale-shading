"""Load dataset views with a disk cache."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import os

import mitsuba as mi
import torch
import tqdm

from datasets import Dataset, Measurement, Measurements

from .common import DataLoader


class ViewDataLoader(DataLoader):
    """Load the first n views (all by default), caching by dataset configuration."""

    def __init__(self, dataset: Dataset, n: int | None = None) -> None:
        self.dataset = dataset
        self.n = n if n is not None else len(dataset)

    def load(self) -> Measurements:
        """Load and pack the selected views as Measurements."""
        scene_dicts: list[dict] = []
        images: list = []

        it = tqdm.tqdm(range(self.n))
        h = self.hash()
        it.set_description(f"Loading Measurements {h}")

        for i in it:
            d = self._load_raw(i)
            scene_dicts.append(d["scene_dict"])
            images.append(d["image"])

        return Measurements.from_raw(scene_dicts, images)

    def _cache_dir(self) -> Path:
        """Cache directory in the dataset folder."""
        return Path(self.dataset.path).parent / ".cache" / self.hash()

    def _load_raw(self, index: int) -> dict:
        """Load a cached view, rebuilding it if the cache is missing or unreadable."""
        cache = self._cache_dir()
        os.makedirs(cache, exist_ok=True)
        cache_pkl = cache / f"{index}.pkl"
        cache_exr = cache / f"{index}.exr"

        if cache_pkl.exists():
            try:
                return torch.load(cache_pkl, weights_only=False)
            except Exception:
                os.remove(cache_pkl)

        m = self.dataset[index]
        d = m.save()
        torch.save(d, cache_pkl)
        mi.util.write_bitmap(str(cache_exr), m.image)
        return d

    def hash(self):
        key = f"{type(self.dataset)}({str(self.dataset.config)}){self.dataset}"
        return sha256(key.encode()).hexdigest()[:16]

    def __getitem__(self, index: int) -> Measurement:
        """View ``index`` as a Measurement."""
        return Measurement.load(self._load_raw(index))

    def __len__(self) -> int:
        return self.n
