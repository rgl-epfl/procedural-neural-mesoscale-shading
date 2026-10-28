"""The style loss of the detail stage."""

from __future__ import annotations

import torch
import torch.nn as torchnn

from losses.style import FeatureModel


class TorchStyleLoss(torchnn.Module):
    """Style loss on image blocks: a transport distance between the VGG features
    of the rendered and the captured blocks, per exposure.
    """

    def __init__(self, loss, mean=True, exposures=None):
        super().__init__()
        self.feature_model = FeatureModel().to("cuda")
        self.loss = loss
        self.mean = mean
        self.exposures = exposures if exposures is not None else [0.0]

    def forward(self, res, L_ref, Le, shape):
        """Args:
        res: BSDF output, with ``f`` [N, 3].
        L_ref: captured radiance [N, 3].
        Le: emitter radiance [N, 3] or [1].
        shape: [n_channels, n_blocks, height, width].
        """
        n_channels, n_blocks, height, width = shape
        pred = res["f"] * Le
        target = L_ref

        # [N, 3] → [n_blocks, 3, h, w]
        pred = pred.reshape(n_blocks, height, width, n_channels).permute(0, 3, 1, 2)
        target = target.reshape(n_blocks, height, width, n_channels).permute(0, 3, 1, 2)

        loss = torch.tensor(0.0, device=pred.device)
        for exposure in self.exposures:
            pred_exp = torch.clamp(pred * 2**exposure, 0, 1)
            target_exp = torch.clamp(target * 2**exposure, 0, 1)

            # gamma 2.2 before VGG, which was trained on sRGB images; the
            # clamp keeps the gradient of x^(1/2.2) finite at 0
            eps = 1e-6
            pred_srgb = pred_exp.clamp(min=eps).pow(1.0 / 2.2)
            target_srgb = target_exp.clamp(min=eps).pow(1.0 / 2.2)

            with torch.no_grad():
                ref_features = self.feature_model(target_srgb)
            res_features = self.feature_model(pred_srgb)

            loss = loss + self.loss(res_features, ref_features).sum()

        if self.mean:
            loss = loss / n_blocks

        return loss
