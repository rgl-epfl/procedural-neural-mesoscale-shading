"""Base type of the data loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from datasets import Measurement, Measurements


class DataLoader(ABC):
    """Base type of the data loaders built from config."""

    @abstractmethod
    def load(self) -> Measurements:
        """All views packed as Measurements."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Measurement:
        """View ``index`` as a Measurement."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
