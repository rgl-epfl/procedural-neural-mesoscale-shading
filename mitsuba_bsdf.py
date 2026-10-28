"""Mitsuba BSDF plugin for rendering trained material checkpoints.

Use ``{"type": "mesoscale", "chkpt": "path/to/checkpoint.pkl"}`` as a BSDF
scene entry. Neural decoders use fp16 cooperative vectors."""

from __future__ import annotations

import pickle

import drjit as dr
import mitsuba as mi
import numpy as np
from drjit import nn
from drjit.auto.ad import (
    ArrayXf,
    ArrayXf16,
    Float,
    Float16,
    Int,
    TensorXf16,
    Texture2f16,
)


def _jit_backend():
    return (
        dr.JitBackend.LLVM
        if (mi.variant() or "").startswith("llvm_")
        else dr.JitBackend.CUDA
    )


def _next_pow2(n: int) -> int:
    return 1 << max(n, 1).bit_length()


_types: dict[str, type] = {}


def _register(cls):
    _types[cls.__name__] = cls
    return cls


def _build(cfg, **kwargs):
    """Build the Mitsuba counterpart of a configured PyTorch model."""
    cls = _types[cfg["_target_"].rsplit(".", 1)[-1].removeprefix("Torch")]
    return cls(**{k: v for k, v in cfg.items() if not k.startswith("_")}, **kwargs)


@_register
class MLP(mi.Object):
    """ReLU decoder evaluated with fp16 cooperative vectors."""

    def __init__(self, n_inputs, n_outputs, width=32, n_hidden=2, bias=True, **_):
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.n_inputs_padded = _next_pow2(self.n_inputs)
        layers = [nn.Cast(Float16)]
        if n_hidden >= 0:
            layers += [nn.Linear(-1, width, bias=bias), nn.ReLU()]
        for _ in range(n_hidden):
            layers += [nn.Linear(-1, -1, bias=bias), nn.ReLU()]
        layers.append(nn.Linear(-1, _next_pow2(self.n_outputs), bias=bias))
        net = nn.Sequential(*layers).alloc(
            TensorXf16, self.n_inputs_padded, rng=dr.rng(seed=42)
        )
        self.weights, self.net = nn.pack(net, layout="training")

    def __call__(self, x):
        def f(x):
            x = nn.CoopVec(x)
            if len(x) != self.n_inputs_padded:
                x = nn.CoopVec(
                    *x, [x.type(0.0)] * (self.n_inputs_padded - self.n_inputs)
                )
            return list(self.net(nn.CoopVec(x)))[: self.n_outputs]

        return dr.func(f, backend=_jit_backend())(list(x))

    def load_weights(self, weights: list[np.ndarray], biases: list[np.ndarray]):
        """Per-layer ``[out, in]`` weights and ``[out]`` biases, zero-padded."""
        net = nn.unpack(self.net, self.weights)[1]
        layers = [layer for layer in net if isinstance(layer, nn.Linear)]
        for layer, W, b in zip(layers, weights, biases):
            n_out, n_in = layer.weights.shape
            Wp = np.zeros((n_out, n_in), dtype=np.float16)
            Wp[: W.shape[0], : W.shape[1]] = W
            bp = np.zeros(n_out, dtype=np.float16)
            bp[: b.shape[0]] = b
            layer.weights, layer.bias = TensorXf16(Wp), TensorXf16(bp)
        self.weights, self.net = nn.pack(net, layout="training")


class Texture:
    n_outputs: int = 0


@_register
class Filtered2D(Texture):
    """Latent texture: n_channels values per texel, bilinear, clamped at the border."""

    def __init__(self, resolution=512, n_channels=1, **_):
        self.n_outputs = n_channels
        self.level = Texture2f16(
            TensorXf16(np.zeros((resolution, resolution, n_channels), np.float16)),
            wrap_mode=dr.WrapMode.Clamp,
        )

    def load(self, level: np.ndarray):
        """``level`` as stored by PyTorch, [1, C, H, W]."""
        self.level = Texture2f16(
            TensorXf16(
                np.ascontiguousarray(level[0].transpose(1, 2, 0)).astype(np.float16)
            ),
            wrap_mode=dr.WrapMode.Clamp,
        )

    def eval(self, uv):
        return ArrayXf16(self.level.eval(uv))


@_register
class PerlinLayers4D(Texture):
    """n_levels octaves of n_channels Perlin noise in (u, v, h.x, h.y)."""

    def __init__(
        self,
        resolution=512,
        n_channels=8,
        n_levels=8,
        const=False,
        dir_freq_ratio=0.25,
        **_,
    ):
        self.n_channels = n_channels
        self.n_levels = n_levels + (1 if const else 0)
        self.const = const
        self.dir_freq_ratio = dir_freq_ratio
        self.base_freq = resolution / (2.0 ** (n_levels - 1))
        self.n_outputs = self.n_channels * self.n_levels
        self.grad = [
            Float([1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]),
            Float([1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1]),
            Float([1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0]),
            Float([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]),
        ]

    @staticmethod
    def _hash4(ix, iy, iz, iw):
        """Gradient index in [0, 16), in signed 32-bit arithmetic as the PyTorch
        version (``>>`` shifts in the sign bit)."""
        I = mi.Int32
        h = ix * I(1619) + iy * I(31337) + iz * I(6971) + iw * I(1013)
        h = h ^ (h >> 8)
        h = h * I(0x21F0AAAD)
        h = h ^ (h >> 15)
        h = h * I(0x735A2D97)
        h = h ^ (h >> 15)
        return mi.UInt32(h & I(15))

    def _noise(self, p):
        g = [dr.floor(c) for c in p]
        f = [c - gc for c, gc in zip(p, g)]
        fade = [x * x * x * (x * (x * 6.0 - 15.0) + 10.0) for x in f]
        i = [mi.Int32(gc) for gc in g]
        corners = []
        for dw in (0, 1):
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        d = (dx, dy, dz, dw)
                        gi = self._hash4(*[i[k] + mi.Int32(d[k]) for k in range(4)])
                        corners.append(
                            sum(
                                dr.gather(Float, self.grad[k], gi) * (f[k] - d[k])
                                for k in range(4)
                            )
                        )

        def lerp(a, b, t):
            return a + t * (b - a)

        c, (u, v, s, t) = corners, fade
        x00 = lerp(lerp(c[0], c[1], u), lerp(c[2], c[3], u), v)
        x10 = lerp(lerp(c[4], c[5], u), lerp(c[6], c[7], u), v)
        x01 = lerp(lerp(c[8], c[9], u), lerp(c[10], c[11], u), v)
        x11 = lerp(lerp(c[12], c[13], u), lerp(c[14], c[15], u), v)
        return lerp(lerp(x00, x10, s), lerp(x01, x11, s), t)

    def eval(self, uv, h):
        out = []
        for level in range(self.n_levels):
            if self.const and level == 0:
                out += [dr.ones(Float, dr.width(uv.x))] * self.n_channels
                continue
            freq = self.base_freq * 2.0 ** (level - (1 if self.const else 0))
            dir_freq = freq * self.dir_freq_ratio
            for ch in range(self.n_channels):
                out.append(
                    self._noise(
                        (
                            uv.x * freq + ch * 43758.5453,
                            uv.y * freq + ch * 12345.6789,
                            h.x * dir_freq + ch * 78932.1234,
                            h.y * dir_freq + ch * 56789.4321,
                        )
                    )
                )
        return out


@_register
class NeuralOffset(Texture):
    """Per-texel parallax: uv moves by depth / cos(theta_i) * wi.xy."""

    def __init__(self, encoding, decoder, directional=True, min_cos=0.01, **_):
        self.directional = directional
        self.min_cos = min_cos
        self.encoding = _build(encoding)
        self.decoder = _build(
            decoder, n_inputs=3 + self.encoding.n_outputs, n_outputs=1
        )

    def eval(self, si):
        wi = si.wi if self.directional else dr.zeros(mi.Vector3f)
        depth = (
            ArrayXf(self.decoder(nn.CoopVec(wi, self.encoding.eval(si.uv))))[0] * 0.1
        )
        return depth / dr.maximum(si.wi.z, self.min_cos) * mi.Point2f(si.wi.x, si.wi.y)


@_register
class NeuralBSDF:
    """Coarse neural BSDF: latent texture and local (wi, wo) through an MLP."""

    def __init__(self, encoding, decoder, **_):
        self.encoding = _build(encoding)
        self.decoder = _build(
            decoder, n_inputs=self.encoding.n_outputs + 6, n_outputs=3
        )

    def coarse(self, enc, wi, wo):
        return dr.exp(mi.Color3f(self.decoder(nn.CoopVec(enc, wi, wo))))

    def eval(self, si, wo):
        return self.coarse(self.encoding.eval(si.uv), si.wi, wo)


@_register
class PerlinDetail(NeuralBSDF):
    """f = f_coarse * f_detail, f_detail = exp(MLP) of 4D Perlin noise."""

    def __init__(self, encoding, decoder, noise_4d, noise_decoder, **_):
        super().__init__(encoding, decoder)
        self.noise_4d = _build(noise_4d)
        n_in = 6 + self.encoding.n_outputs + self.noise_4d.n_outputs
        self.noise_decoder = _build(noise_decoder, n_inputs=n_in, n_outputs=3)

    def eval(self, si, wo):
        enc = self.encoding.eval(si.uv)
        noise = self.noise_4d.eval(si.uv, dr.normalize(si.wi + wo))
        detail = dr.exp(
            mi.Color3f(self.noise_decoder(nn.CoopVec(enc, si.wi, wo, noise)))
        )
        return self.coarse(enc, si.wi, wo) * detail


@_register
class Coarse:
    """Parallax offset, then the BSDF."""

    def __init__(self, bsdf, offset=None, **_):
        self.offset = _build(offset) if offset else None
        self.bsdf = _build(bsdf)

    def eval(self, si, wo):
        if self.offset is not None:
            si = mi.SurfaceInteraction3f(si)
            si.uv += self.offset.eval(si)
        return self.bsdf.eval(si, wo)


_register(type("ParallaxDetail", (Coarse,), {"__doc__": "Detail-stage BSDF with a frozen parallax offset."}))

# Checkpoints saved before the rename embed the old Hydra target names.
_types["NeuMIP"] = Coarse
_types["NeuMIPDetail"] = _types["ParallaxDetail"]


def _load(model, state: dict):
    """Load checkpoint weights into the model."""

    def obj(path):
        o = model
        for part in path.split("."):
            o = getattr(o, part)
        return o

    for key in state:
        if key.endswith(".net.0.weight"):
            prefix = key.removesuffix(".net.0.weight")
            n = sorted(
                {
                    int(k.split(".")[-2])
                    for k in state
                    if k.startswith(prefix + ".net.") and k.endswith(".weight")
                }
            )
            obj(prefix).load_weights(
                [np.asarray(state[f"{prefix}.net.{i}.weight"], np.float32) for i in n],
                [np.asarray(state[f"{prefix}.net.{i}.bias"], np.float32) for i in n],
            )
        elif key.endswith(".level") and not key.endswith("._encoding.level"):
            obj(key.removesuffix(".level")).load(np.asarray(state[key], np.float32))


class MesoscaleBSDF(mi.BSDF):
    """Mitsuba plugin ``mesoscale``.

    Properties: ``chkpt`` (training checkpoint), ``norm_scale`` (radiance
    scale, default 1), ``uv_scale_u/v``, ``uv_offset_u/v`` (default identity),
    ``uv_mirror`` (mirrored repeat outside [0, 1], default false) and
    ``uv_inset`` (shrinks [0, 1] by this margin on each side, default 0).
    Sampling is cosine-weighted.
    """

    def __init__(self, props: mi.Properties):
        super().__init__(props)
        with open(props.get("chkpt"), "rb") as f:
            chkpt = pickle.load(f)
        state = {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v)
            for k, v in chkpt["bsdf_state"].items()
        }
        self.model = _build(chkpt["bsdf_cfg"])
        _load(self.model, state)
        self.norm_scale = float(props.get("norm_scale", 1.0))
        self.uv_scale = (
            float(props.get("uv_scale_u", 1.0)),
            float(props.get("uv_scale_v", 1.0)),
        )
        self.uv_offset = (
            float(props.get("uv_offset_u", 0.0)),
            float(props.get("uv_offset_v", 0.0)),
        )
        self.uv_mirror = bool(props.get("uv_mirror", False))
        self.uv_inset = float(props.get("uv_inset", 0.0))
        self.m_components = [mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide]
        self.m_flags = self.m_components[0]

    def _uv(self, si):
        u = si.uv.x * self.uv_scale[0] + self.uv_offset[0]
        v = si.uv.y * self.uv_scale[1] + self.uv_offset[1]
        if self.uv_mirror:

            def mirror(x):
                t = x - dr.floor(x)
                return dr.select((Int(dr.floor(x)) & 1) == 1, 1.0 - t, t)

            u, v = mirror(u), mirror(v)
        if self.uv_inset > 0:
            s = 1.0 - 2.0 * self.uv_inset
            u, v = u * s + self.uv_inset, v * s + self.uv_inset
        si = mi.SurfaceInteraction3f(si)
        si.uv = mi.Point2f(u, v)
        return si

    def f(self, si, wo, active=True):
        """The BSDF value (without the cosine), zero below the surface."""
        f = self.model.eval(self._uv(si), wo) * self.norm_scale
        active = mi.Bool(active) & (si.wi.z > 0) & (wo.z > 0)
        return dr.select(active, dr.maximum(f, 0.0), 0.0)

    def eval(self, ctx, si, wo, active=True):
        return self.f(si, wo, active) * dr.maximum(mi.Frame3f.cos_theta(wo), 0.0)

    def pdf(self, ctx, si, wo, active=True):
        ok = mi.Bool(active) & (si.wi.z > 0) & (wo.z > 0)
        return dr.select(ok, mi.warp.square_to_cosine_hemisphere_pdf(wo), 0.0)

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def sample(self, ctx, si, sample1, sample2, active=True):
        bs = dr.zeros(mi.BSDFSample3f)
        bs.wo = mi.warp.square_to_cosine_hemisphere(sample2)
        bs.pdf = mi.warp.square_to_cosine_hemisphere_pdf(bs.wo)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DiffuseReflection)
        bs.sampled_component = 0
        return bs, self.eval(ctx, si, bs.wo, active) / dr.maximum(bs.pdf, 1e-8)

    def to_string(self):
        return "MesoscaleBSDF[]"


mi.register_bsdf("mesoscale", lambda props: MesoscaleBSDF(props))
