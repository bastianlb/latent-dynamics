"""Small encoder + linear-probe model for studying latent class structure.

Default architecture: an MLP encoder (in_channels*img_size*img_size -> 256
-> 64 -> latent_dim, ReLU) feeding a linear probe head (latent_dim -> 10). A
tiny CNN encoder is available via `encoder_arch`, generalized to accept
arbitrary (in_channels, img_size) -- e.g. 1x28x28 for mnist/fashion_mnist or
3x32x32 for cifar10 -- by computing its flattened conv-output size
dynamically at construction time (see CNNEncoder). An optional decoder
(latent_dim -> in_channels*img_size*img_size) is built whenever `objective`
needs reconstruction.

INPUT SHAPE PLUMBING (in_channels / img_size): `LatentProbeModel.__init__`
accepts optional `in_channels`/`img_size` kwargs. train.py's model
construction call site is out of scope for the dataset-generalization
change this module is part of, and it only ever passes
(latent_dim, encoder_arch, objective, latent_norm) -- it does NOT pass
in_channels/img_size explicitly. So when omitted, they fall back to this
module's mutable defaults `_DEFAULT_IN_CHANNELS`/`_DEFAULT_IMG_SIZE`
(initially the mnist/fashion_mnist shape, 1x28). `data.py`'s `load_mnist()`
calls `set_default_input_shape(cfg.in_channels, cfg.img_size)` as a side
effect, BEFORE train.py constructs the model, so that by construction time
the defaults already reflect `cfg.dataset` -- giving a correctly-shaped
encoder/decoder for cifar10 (3x32x32) etc. without changing train.py's call
signature. Callers that DO have direct access to a resolved Config (e.g.
tests, new scripts) should prefer passing `in_channels=cfg.in_channels,
img_size=cfg.img_size` explicitly rather than relying on this side channel.

THE LATENT: `encode(x)` returns the encoder output AFTER `latent_norm` is
applied. This post-norm latent is what the probe/decoder consume AND what
train.py saves into every `latents/step_*.npz` snapshot (SPEC.md's data
contract) -- i.e. the snapshot always reflects the representation actually
used downstream, not the raw encoder output.

Objectives (`objective`, controls what shapes the encoder via gradients):
  - "probe": loss = CE(probe(latent), y). Probe gradients flow into the
    encoder (this is the original/default behavior).
  - "autoencoder": the encoder (and decoder) are shaped ONLY by
    MSE(decoder(latent), x). The probe is still trained every step as a pure
    linear-separability READOUT, but on `latent.detach()` -- its gradients
    update only the probe's own weights and never reach the encoder. Requires
    a decoder.
  - "probe_recon": loss = CE(probe(latent), y) + recon_weight * MSE(decoder(latent), x).
    Both terms shape the encoder (probe gradients flow through; decoder is
    active). Requires a decoder.
`forward()` always computes and returns probe logits (regardless of
objective), so callers can always compute a CE/accuracy readout.

Latent normalization (`latent_norm`, applied to the raw encoder output to
produce THE latent):
  - "none": identity.
  - "l2": z / (||z||_2 + eps) -- projects onto (near) the unit sphere.
  - "bn": nn.BatchNorm1d(latent_dim) -- standard train/eval-mode batchnorm
    (batch statistics while `model.train()`, running statistics while
    `model.eval()` -- including during latent snapshotting, since
    train.py always calls `model.eval()` before `encode()`).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

N_CLASSES = 10
IMG_SIZE = 28  # legacy default (mnist/fashion_mnist) -- seeds _DEFAULT_IMG_SIZE below
IMG_DIM = IMG_SIZE * IMG_SIZE

OBJECTIVES = ("probe", "autoencoder", "probe_recon")
LATENT_NORMS = ("none", "l2", "bn")

# Mutable module-level defaults for (in_channels, img_size), used by
# LatentProbeModel when constructed without explicit in_channels/img_size
# kwargs. See the module docstring's "INPUT SHAPE PLUMBING" note for why
# this side channel exists (train.py's model-construction call site isn't
# touched by the dataset-generalization change).
_DEFAULT_IN_CHANNELS = 1
_DEFAULT_IMG_SIZE = IMG_SIZE


def set_default_input_shape(in_channels: int, img_size: int) -> None:
    """Set the module-level (in_channels, img_size) default used by
    LatentProbeModel (and build_encoder) when constructed without explicit
    in_channels/img_size kwargs. Called by `data.py`'s `load_mnist()` with
    `cfg.in_channels`/`cfg.img_size` before train.py builds its model -- see
    the module docstring.
    """
    global _DEFAULT_IN_CHANNELS, _DEFAULT_IMG_SIZE
    _DEFAULT_IN_CHANNELS = int(in_channels)
    _DEFAULT_IMG_SIZE = int(img_size)


class MLPEncoder(nn.Module):
    """in_channels*img_size*img_size -> 256 -> 64 -> latent_dim, ReLU
    activations. Input: [N, in_channels, img_size, img_size]. Defaults to
    the classic 1x28x28 (mnist/fashion_mnist) shape.
    """

    def __init__(self, latent_dim: int, in_channels: int = 1, img_size: int = 28) -> None:
        super().__init__()
        img_dim = in_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNEncoder(nn.Module):
    """Tiny conv encoder: two conv+pool blocks then a linear projection to
    latent_dim. Input: [N, in_channels, img_size, img_size] -- generalized
    to work for any (in_channels, img_size) (e.g. 1x28x28 or 3x32x32) by
    computing the flattened conv-output size dynamically via one dummy
    forward pass at construction time. This is a ONE-TIME, fully-materializing
    computation done inside __init__ (plain nn.Conv2d/nn.Linear, built with
    already-known shapes) -- deliberately NOT implemented with nn.Lazy*
    modules, which defer parameter creation until the first real forward
    pass; train.py builds its optimizer immediately after constructing the
    model (before any forward pass), so lazily-initialized parameters would
    never make it into the optimizer's param groups.
    """

    def __init__(self, latent_dim: int, in_channels: int = 1, img_size: int = 28) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # e.g. 28 -> 14, or 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # e.g. 14 -> 7, or 16 -> 8
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            flat_dim = self.conv(dummy).flatten(1).shape[1]
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(flat_dim, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNet18Encoder(nn.Module):
    """Standard CIFAR-style ResNet-18: 3x3 stride-1 stem, no maxpool, 4 stages of 2 BasicBlocks
    (64/128/256/512), global average pool -> 512-d penultimate feature. If latent_dim != 512 a
    linear bottleneck 512 -> latent_dim follows (the output is then fed to latent_norm and the
    linear head, exactly like the other encoders); latent_dim == 512 means no bottleneck.
    """

    FEATURE_DIM = 512

    def __init__(self, latent_dim: int, in_channels: int = 3, img_size: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        layers, in_planes = [], 64
        for planes, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            for s in (stride, 1):
                layers.append(_BasicBlock(in_planes, planes, s))
                in_planes = planes
        self.layers = nn.Sequential(*layers)
        self.bottleneck = nn.Linear(self.FEATURE_DIM, latent_dim) if latent_dim != self.FEATURE_DIM else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(torch.relu(self.bn1(self.conv1(x))))
        return self.bottleneck(torch.flatten(nn.functional.adaptive_avg_pool2d(out, 1), 1))


class LinearEncoder(nn.Module):
    """Single Linear(img_dim -> latent_dim) with bias (flatten first). Used for --linear_ae."""

    def __init__(self, latent_dim: int, in_channels: int = 1, img_size: int = 28) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_channels * img_size * img_size, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearDecoder(nn.Module):
    """Single Linear(latent_dim -> img_dim) with bias. Used for --linear_ae."""

    def __init__(self, latent_dim: int, in_channels: int = 1, img_size: int = 28) -> None:
        super().__init__()
        self.net = nn.Linear(latent_dim, in_channels * img_size * img_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MLPDecoder(nn.Module):
    """latent_dim -> 64 -> 256 -> in_channels*img_size*img_size (mirrors
    MLPEncoder; output stays FLAT, matching train.py's
    `mse_loss(recon, images.flatten(1))` regardless of in_channels/img_size).
    Used only when a decoder is required (objective in {"autoencoder",
    "probe_recon"}). Used for both encoder_arch="mlp" and "cnn" -- there is
    only ever one decoder class, independent of encoder_arch.
    """

    def __init__(self, latent_dim: int, in_channels: int = 1, img_size: int = 28) -> None:
        super().__init__()
        img_dim = in_channels * img_size * img_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, img_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ConvAEEncoder(nn.Module):
    """Opt-in conv AE encoder (encoder_arch='convae'): 3 stride-2 4x4 convs (C->32->64->128, ReLU)
    then Linear(128*(s/8)^2 -> latent_dim). 32x32 input -> 4x4x128."""

    def __init__(self, latent_dim: int, in_channels: int = 3, img_size: int = 32) -> None:
        super().__init__()
        assert img_size % 8 == 0
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(128 * (img_size // 8) ** 2, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class ConvAEDecoder(nn.Module):
    """Mirror of ConvAEEncoder: Linear(d -> 128*(s/8)^2) -> 3 transposed 4x4 stride-2 convs
    (128->64->32->C, ReLU between). Output flattened to [N, C*s*s] like the other decoders."""

    def __init__(self, latent_dim: int, in_channels: int = 3, img_size: int = 32) -> None:
        super().__init__()
        self.s = img_size // 8
        self.fc = nn.Linear(latent_dim, 128 * self.s * self.s)
        self.deconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, 4, 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.deconv(self.fc(z).view(z.shape[0], 128, self.s, self.s)).flatten(1)


def build_encoder(encoder_arch: str, latent_dim: int, in_channels: int = 1, img_size: int = 28) -> nn.Module:
    """Factory for the encoder selected by `Config.encoder_arch`."""
    if encoder_arch == "mlp":
        return MLPEncoder(latent_dim, in_channels=in_channels, img_size=img_size)
    if encoder_arch == "cnn":
        return CNNEncoder(latent_dim, in_channels=in_channels, img_size=img_size)
    if encoder_arch == "convae":
        return ConvAEEncoder(latent_dim, in_channels=in_channels, img_size=img_size)
    if encoder_arch == "resnet18":
        return ResNet18Encoder(latent_dim, in_channels=in_channels, img_size=img_size)
    raise ValueError(f"Unknown encoder_arch: {encoder_arch!r} (expected 'mlp', 'cnn' or 'resnet18')")


class LatentProbeModel(nn.Module):
    """Encoder + linear probe head, optional decoder, selectable objective and
    latent normalization. See module docstring for the full contract.

    - `encode(x)`  -> THE latent only (post latent_norm). Call in `eval()`
      mode under `no_grad()` for latent snapshotting (see SPEC.md).
    - `forward(x)` -> `(probe_logits, recon_or_None)`. `probe_logits` is
      always populated; for `objective="autoencoder"` it is computed from a
      *detached* latent so the probe trains as a readout without shaping the
      encoder.
    """

    _L2_EPS = 1e-8

    def __init__(
        self,
        latent_dim: int,
        encoder_arch: str = "mlp",
        objective: str = "probe",
        latent_norm: str = "none",
        in_channels: Optional[int] = None,
        img_size: Optional[int] = None,
        l2_scale: Optional[float] = None,
        l2_scale_learned: bool = False,
        linear_ae: bool = False,
    ) -> None:
        super().__init__()
        if linear_ae and objective == "probe":
            raise ValueError("linear_ae requires a reconstruction objective")
        if l2_scale is not None and latent_norm != "l2":
            raise ValueError("l2_scale requires latent_norm='l2'")
        if l2_scale_learned and l2_scale is None:
            raise ValueError("l2_scale_learned requires l2_scale")
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective: {objective!r} (expected one of {OBJECTIVES})")
        if latent_norm not in LATENT_NORMS:
            raise ValueError(f"Unknown latent_norm: {latent_norm!r} (expected one of {LATENT_NORMS})")

        # in_channels/img_size default to this module's mutable defaults
        # (see `set_default_input_shape` and the module docstring's "INPUT
        # SHAPE PLUMBING" note) when not passed explicitly.
        resolved_in_channels = _DEFAULT_IN_CHANNELS if in_channels is None else in_channels
        resolved_img_size = _DEFAULT_IMG_SIZE if img_size is None else img_size

        self.latent_dim = latent_dim
        self.objective = objective
        self.latent_norm = latent_norm
        self.in_channels = resolved_in_channels
        self.img_size = resolved_img_size

        if linear_ae:
            self.encoder = LinearEncoder(latent_dim, resolved_in_channels, resolved_img_size)
        else:
            self.encoder = build_encoder(encoder_arch, latent_dim, resolved_in_channels, resolved_img_size)
        self.probe = nn.Linear(latent_dim, N_CLASSES)

        # NormFace-style temperature: the probe sees `scale * z` (z = the unit-norm latent,
        # which is still what encode() returns / snapshots store). None -> scale 1 (original).
        self.l2_scale: Optional[nn.Parameter] = None
        self._l2_scale_const: Optional[float] = None
        if l2_scale is not None:
            if l2_scale_learned:
                self.l2_scale = nn.Parameter(torch.tensor(float(l2_scale)))
            else:
                self._l2_scale_const = float(l2_scale)

        # Registered as a submodule (not None) only when latent_norm == "bn",
        # so its running stats / affine params move with .to(device) and are
        # included in state_dict() exactly when relevant.
        self.bn: Optional[nn.Module] = nn.BatchNorm1d(latent_dim) if latent_norm == "bn" else None

        needs_decoder = objective in ("autoencoder", "probe_recon")
        self.decoder: Optional[nn.Module] = (
            (LinearDecoder if linear_ae else ConvAEDecoder if encoder_arch == "convae" else MLPDecoder)(
                latent_dim, in_channels=resolved_in_channels, img_size=resolved_img_size)
            if needs_decoder else None
        )

    def current_l2_scale(self) -> Optional[float]:
        """Current value of the head scale (None when no l2_scale was configured)."""
        if self.l2_scale is not None:
            return float(self.l2_scale.detach())
        return self._l2_scale_const

    def _apply_latent_norm(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_norm == "none":
            return z
        if self.latent_norm == "l2":
            z = z.float()  # no-op in fp32; keeps the norm in fp32 under bf16 autocast
            return z / z.norm(p=2, dim=1, keepdim=True).clamp_min(self._L2_EPS)
        # latent_norm == "bn"
        return self.bn(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output AFTER latent_norm -- THE latent (saved in snapshots)."""
        return self._apply_latent_norm(self.encoder(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = self.encode(x)
        # For "autoencoder", detach before the probe so probe gradients
        # cannot reach the encoder; the probe still gets its own CE gradient
        # (it is a pure readout trained alongside, not shaping the latent).
        z_for_probe = z.detach() if self.objective == "autoencoder" else z
        if self.l2_scale is not None:
            z_for_probe = z_for_probe * self.l2_scale
        elif self._l2_scale_const is not None:
            z_for_probe = z_for_probe * self._l2_scale_const
        logits = self.probe(z_for_probe)
        recon = self.decoder(z) if self.decoder is not None else None
        return logits, recon
