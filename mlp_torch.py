"""MLP decoder for neural materials."""

from __future__ import annotations


import torch
import torch.nn as torchnn


class TorchMLP(torchnn.Module):
    """ReLU network: n_hidden + 1 hidden layers of the given width (n_hidden = -1: a linear map)."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        width: int = 32,
        n_hidden: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        layers = []
        in_size = n_inputs
        if n_hidden >= 0:
            layers.append(torchnn.Linear(in_size, width, bias=bias))
            layers.append(torchnn.ReLU())
            in_size = width
        for _ in range(n_hidden):
            layers.append(torchnn.Linear(in_size, width, bias=bias))
            layers.append(torchnn.ReLU())
            in_size = width
        layers.append(torchnn.Linear(in_size, n_outputs, bias=bias))

        self.net = torchnn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, torchnn.Linear):
                torchnn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torchnn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
