"""DrJit to PyTorch conversions, for rendering."""

import torch
from drjit.auto.ad import TensorXf


def vec3_to_torch(v):
    """DrJit Vector3f to a [n, 3] tensor, on the GPU."""
    return torch.stack(
        [TensorXf(v.x).torch(), TensorXf(v.y).torch(), TensorXf(v.z).torch()], dim=1
    )


def color3_to_torch(c):
    """DrJit Color3f to a [n, 3] tensor, on the GPU."""
    return torch.stack(
        [TensorXf(c.x).torch(), TensorXf(c.y).torch(), TensorXf(c.z).torch()], dim=1
    )
