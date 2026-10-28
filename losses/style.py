"""VGG features and the relaxed optimal-transport distance used by the style loss."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import vgg16


class FeatureModel(nn.Module):
    """Activations of a pretrained VGG16 at ``layers``."""

    def __init__(self, layers: tuple[int, ...] = (1, 6, 11, 18, 25)) -> None:
        super().__init__()
        self.layers = list(layers)
        self.vgg = vgg16(weights="IMAGENET1K_V1").features.eval()
        self.vgg.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
        x = (x - mean) / std

        features = []

        max_layer = max(*self.layers)
        for i, layer in enumerate(self.vgg.children()):
            x = layer(x)
            if i in self.layers:
                features.append(x)
            if i > max_layer:
                break

        return features


class RelaxedOTLoss(nn.Module):
    """Relaxed optimal transport (Kolkin et al. 2019, arXiv:1904.12785) with the
    cosine distance, as in https://github.com/IVRL/NoiseNCA/blob/main/loss.py#L174.
    """

    def __init__(self, n_samples: int = 1024, momentum: bool = True) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.momentum = momentum

    def distance(self, x, y):
        x_norm = torch.norm(x, dim=2, keepdim=True)  # (b, n, 1)
        y_t = y.transpose(1, 2)  # (b, c, m) (m may be different from n)
        y_norm = torch.norm(y_t, dim=1, keepdim=True)  # (b, 1, m)
        dist = 1.0 - torch.matmul(x, y_t) / (x_norm * y_norm + 1e-10)  # (b, n, m)
        return dist

    def style_loss(self, x, y):
        pairwise_distance = self.distance(x, y)
        m1, m1_inds = pairwise_distance.min(1)
        m2, m2_inds = pairwise_distance.min(2)
        remd = torch.max(m1.mean(dim=1), m2.mean(dim=1))
        return remd

    def moment_loss(self, x, y):
        mu_x, mu_y = torch.mean(x, 1, keepdim=True), torch.mean(y, 1, keepdim=True)
        mu_diff = torch.abs(mu_x - mu_y).mean(dim=(1, 2))

        x_c, y_c = x - mu_x, y - mu_y
        x_cov = torch.matmul(x_c.transpose(1, 2), x_c) / (x.shape[1] - 1)
        y_cov = torch.matmul(y_c.transpose(1, 2), y_c) / (y.shape[1] - 1)

        cov_diff = torch.abs(x_cov - y_cov).mean(dim=(1, 2))
        return mu_diff + cov_diff

    def forward(self, target_features, source_features):
        loss = 0.0
        for x, y in zip(source_features, target_features):
            b, c, h, w = x.shape
            x, y = [t.reshape(b, c, w * h) for t in (x, y)]
            (b_x, c, n_x), (b_y, _, n_y) = x.shape, y.shape
            n_samples = min(n_x, n_y, self.n_samples)
            indices_x = torch.argsort(torch.rand(b_x, 1, n_x, device=x.device), dim=-1)[
                ..., :n_samples
            ]
            x = x.gather(-1, indices_x.expand(b_x, c, n_samples))
            indices_y = torch.argsort(torch.rand(b_y, 1, n_y, device=y.device), dim=-1)[
                ..., :n_samples
            ]
            y = y.gather(-1, indices_y.expand(b_y, c, n_samples))
            x, y = x.transpose(1, 2), y.transpose(1, 2)  # (b, n_samples, c)
            loss += self.style_loss(x, y)
            if self.momentum:
                loss += self.moment_loss(x, y)

        return loss
