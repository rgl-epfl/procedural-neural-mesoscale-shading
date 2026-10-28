"""Neural BSDFs with learned parallax and Perlin detail."""

from __future__ import annotations

import torch
import torch.nn as torchnn

torch.set_float32_matmul_precision("high")


class TorchNeuralBSDF(torchnn.Module):
    """Decode a latent texture and local directions into a coarse BSDF."""

    def __init__(self, encoding, decoder):
        super().__init__()
        self.encoding = encoding
        self.add_module("_encoding", self.encoding)
        self.decoder = decoder(n_inputs=self.encoding.n_outputs + 6, n_outputs=3)

        self._compiled = False

    def compile(self):
        """JIT-compile the forward pass for kernel fusion."""
        if not self._compiled:
            self.forward = torch.compile(self.forward)
            self._compiled = True

    def prepare(self, it):
        if hasattr(self.encoding, "prepare"):
            self.encoding.prepare(it)

    def eval_bsdf(self, enc, wi, wo):
        return torch.exp(self.decoder(torch.cat([enc, wi, wo], dim=1)))

    def forward(self, si_uv, wi, wo, **kwargs):
        enc = self.encoding(si_uv)

        f = self.eval_bsdf(enc, wi, wo)

        active = ((wi[:, 2:3] > 0) & (wo[:, 2:3] > 0)).float()
        return dict(f=f * active)


class TorchDetail(TorchNeuralBSDF):
    """Scale a frozen coarse BSDF by a learned, positive detail factor."""

    def eval_detail(self, enc, si_uv, wi, wo):
        raise NotImplementedError

    def forward(self, si_uv, wi, wo, **kwargs):
        with torch.no_grad():
            enc = self.encoding(si_uv)
            f_coarse = self.eval_bsdf(enc, wi, wo)

        result_detail = self.eval_detail(enc, si_uv, wi, wo)
        f_detail = result_detail["f"]

        f = f_coarse * f_detail

        return dict(
            f=f,
            f_detail=f_detail,
            f_coarse=f_coarse,
        )


class TorchPerlinDetail(TorchDetail):
    """Model spatial and directional detail with 4D Perlin noise."""

    def __init__(self, encoding, decoder, noise_4d, noise_decoder):
        super().__init__(encoding=encoding, decoder=decoder)
        self.noise_4d = noise_4d
        n_inputs = 6 + self.encoding.n_outputs + self.noise_4d.n_outputs
        self.noise_decoder = noise_decoder(n_inputs=n_inputs, n_outputs=3)

    def eval_detail(self, enc, si_uv, wi, wo):
        h = torchnn.functional.normalize(wi + wo, dim=1)
        noise = self.noise_4d(si_uv, h)
        f = self.noise_decoder(torch.cat([enc, wi, wo, noise], dim=1))
        return dict(f=torch.exp(f))


class TorchNeuralOffset(torchnn.Module):
    """Predict a parallax offset from the latent texture and incident direction."""

    def __init__(self, encoding, decoder, directional=True, min_cos=0.01):
        super().__init__()
        self.directional = directional
        self.min_cos = min_cos
        self._compiled = False
        self.encoding = encoding
        self.add_module("_encoding", self.encoding)
        n_inputs = 3 + self.encoding.n_outputs
        self.decoder = decoder(n_inputs=n_inputs, n_outputs=1)

    def compile(self):
        if not self._compiled:
            self.forward = torch.compile(self.forward)
            self._compiled = True

    def prepare(self, it):
        if hasattr(self.encoding, "prepare"):
            self.encoding.prepare(it)

    def eval_depth(self, si_uv, wi):
        enc = self.encoding(si_uv)
        wi_input = wi if self.directional else torch.zeros_like(wi)
        r = self.decoder(torch.cat([wi_input, enc], dim=1)) * 0.1
        return r

    def forward(self, si_uv, wi):
        r = self.eval_depth(si_uv, wi)
        wiz = torch.clamp(wi[:, 2:3], min=self.min_cos)
        return r / wiz * wi[:, :2]


class TorchCoarse(torchnn.Module):
    """Our coarse BSDF, inspired by NeuMIP (Kuznetsov et al., 2021).

    Parallax is trained during [offset_start, offset_stop).
    Reference: https://doi.org/10.1145/3450626.3459795
    """

    def __init__(self, bsdf, offset=None, offset_start=0, offset_stop=None):
        super().__init__()

        self.offset = offset
        self.offset_start = offset_start
        self.offset_stop = offset_stop
        self.bsdf = bsdf

    def compile(self):
        self.bsdf.compile()
        if self.offset is not None:
            self.offset.compile()

    def prepare(self, it):
        if self.offset is not None:
            self.offset.prepare(it)
        self.bsdf.prepare(it)

    def forward(self, si_uv, wi, wo, it=-1, **kwargs):
        if self.offset is not None:
            si_uv = si_uv + self.offset(si_uv, wi)
            if it < self.offset_start:
                si_uv = si_uv.detach()
            if self.offset_stop is not None and it >= self.offset_stop:
                si_uv = si_uv.detach()
        return self.bsdf(si_uv, wi, wo)


class TorchParallaxDetail(torchnn.Module):
    """Detail-stage BSDF with a frozen parallax offset."""

    def __init__(self, bsdf, offset=None):
        super().__init__()
        self.offset = offset
        self.bsdf = bsdf

    def compile(self):
        self.bsdf.compile()
        if self.offset is not None:
            self.offset.compile()

    def prepare(self, it):
        if self.offset is not None:
            self.offset.prepare(it)
        self.bsdf.prepare(it)

    def forward(self, si_uv, wi, wo, it=-1, **kwargs):
        if self.offset is not None:
            with torch.no_grad():
                si_uv = si_uv.detach() + self.offset(si_uv.detach(), wi.detach())
            si_uv = si_uv.detach()
        return self.bsdf(si_uv, wi, wo)
