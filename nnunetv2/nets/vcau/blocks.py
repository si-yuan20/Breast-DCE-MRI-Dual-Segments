"""Building blocks for VCAU-Net v2.

Everything here is strictly 3D and plans-driven: kernels, normalisation and
channel counts all come from the nnU-Net plans.

Reuse policy
------------
The v1 package (``nnunetv2.nets.vcau``) is *not* modified. Three of its
components are imported and reused because they encode contracts that are
already verified by tests and must not be re-derived:

* ``PhaseSpecificStem3D`` -- the feature of the method, unchanged in v2.
* ``ConvNormAct3D``       -- ``Conv3d -> norm -> GELU`` with odd-kernel padding.
* ``Warp3D`` / ``build_identity_grid`` -- the empirically calibrated
  ``grid_sample`` coordinate contract (see v1's ``spca.py``).

What v2 replaces is everything to do with *capacity*: the fusion operators here
are depthwise-separable and roughly 5x lighter than v1's, because v1's
SPCA+UGCF accounted for ~18.4M of a ~49.8M model and made the ablation arms
capacity-confounded.

Tensor layout follows nnU-Net: ``[B, C, D, H, W]``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Type

import torch
from torch import nn

from nnunetv2.nets.vcau.blocks import (  # re-exported for the rest of vcau_v2
    ConvNormAct3D,
    PhaseSpecificStem3D,
    _validate_kernel_size as validate_kernel_size,
)

__all__ = [
    "ConvNormAct3D",
    "PhaseSpecificStem3D",
    "validate_kernel_size",
    "DepthwiseSeparableBlock3D",
    "LightweightCrossPhaseFusion3D",
    "ContextBranch3D",
]


def _norm(norm_op: Optional[Type[nn.Module]], channels: int, kwargs: Optional[dict]) -> nn.Module:
    return norm_op(channels, **(kwargs or {})) if norm_op is not None else nn.Identity()


class DepthwiseSeparableBlock3D(nn.Module):
    """``1x1x1 reduce -> k^3 depthwise -> 1x1x1 project``, each with norm, final GELU.

    This is the light replacement for v1's ``3C -> C`` dense stack. The depthwise
    stage carries the spatial mixing at ``28C`` parameters instead of ``27C^2``,
    which is what keeps UGCFv2 affordable at the 256- and 320-channel stages.

    Args:
        conv_op: plans convolution class (must be ``nn.Conv3d``).
        in_channels: input width (``3*channels`` for the UGCFv2 evidence tensor).
        out_channels: output width.
        kernel_size: 3-tuple of odd ints, from the plans.
        mid_channels: width of the depthwise stage; defaults to ``out_channels``.
        norm_op / norm_op_kwargs: plans normalisation.
    """

    def __init__(
        self,
        conv_op: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if conv_op is not nn.Conv3d:
            raise ValueError(
                f"VCAU-Net v2 only supports 3D convolutions, got "
                f"{getattr(conv_op, '__name__', conv_op)}."
            )
        kernel_size = validate_kernel_size(kernel_size)
        mid = int(mid_channels) if mid_channels is not None else int(out_channels)
        if mid <= 0:
            raise ValueError(f"mid_channels must be positive, got {mid}.")

        self.reduce = ConvNormAct3D(
            conv_op, in_channels, mid, (1, 1, 1), norm_op, norm_op_kwargs
        )
        padding = tuple(k // 2 for k in kernel_size)
        self.depthwise = conv_op(mid, mid, kernel_size, 1, padding, groups=mid, bias=False)
        self.depthwise_norm = _norm(norm_op, mid, norm_op_kwargs)
        self.project = ConvNormAct3D(
            conv_op, mid, out_channels, (1, 1, 1), norm_op, norm_op_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = torch.nn.functional.gelu(self.depthwise_norm(self.depthwise(x)))
        return self.project(x)


class LightweightCrossPhaseFusion3D(nn.Module):
    """Stage-wise fusion for the stages that carry no alignment and no reliability.

    ``F_avg = 0.5*(F2 + F5)``, ``Delta = |F2 - F5|``, ``F = proj([F_avg, Delta]) + F_avg``.

    The residual form means the operator only has to learn a *correction* to the
    plain phase average rather than reproduce its magnitude, which is why the
    projection can be a single ``1x1x1`` convolution.

    Args:
        channels: feature width of both phases.
        hidden_ratio: capacity knob. ``1.0`` gives the single-``1x1x1`` form
            (2C^2 parameters); larger values insert a hidden ``GELU`` bottleneck
            (``2C*h + hC`` parameters). Used by the capacity-matched control arm
            so that "more parameters" can be ruled out as the cause of a gain.
        out_channels: output width; defaults to ``channels``.
    """

    def __init__(
        self,
        conv_op: Type[nn.Module],
        channels: int,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        hidden_ratio: float = 1.0,
        out_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if conv_op is not nn.Conv3d:
            raise ValueError("VCAU-Net v2 only supports 3D convolutions.")
        if hidden_ratio <= 0:
            raise ValueError(f"hidden_ratio must be positive, got {hidden_ratio}.")

        out_channels = int(out_channels) if out_channels is not None else int(channels)
        self.channels = int(channels)
        self.out_channels = out_channels
        self.hidden_ratio = float(hidden_ratio)
        hidden = max(1, int(round(self.hidden_ratio * self.channels))) if hidden_ratio > 1.0 else None

        if hidden is None:
            self.project: nn.Module = nn.Sequential(
                conv_op(2 * self.channels, out_channels, 1, 1, 0, bias=False),
                _norm(norm_op, out_channels, norm_op_kwargs),
            )
        else:
            self.project = nn.Sequential(
                conv_op(2 * self.channels, hidden, 1, 1, 0, bias=False),
                _norm(norm_op, hidden, norm_op_kwargs),
                nn.GELU(),
                conv_op(hidden, out_channels, 1, 1, 0, bias=False),
                _norm(norm_op, out_channels, norm_op_kwargs),
            )
        self.residual = (
            nn.Identity()
            if out_channels == self.channels
            else conv_op(self.channels, out_channels, 1, 1, 0, bias=False)
        )

    def forward(self, f2: torch.Tensor, f5: torch.Tensor) -> torch.Tensor:
        if f2.shape != f5.shape:
            raise ValueError(
                f"Phase features must match, got {tuple(f2.shape)} vs {tuple(f5.shape)}."
            )
        average = 0.5 * (f2 + f5)
        delta = (f2 - f5).abs()
        fused = self.project(torch.cat((average, delta), dim=1)) + self.residual(average)
        return fused


class ContextBranch3D(nn.Module):
    """Local context integrator used when the fusion gate does not trust the phases.

    Deliberately depthwise: ``k^3`` depthwise convolution over the concatenated
    phase pair followed by a ``1x1x1`` projection to ``channels``. At the widest
    stage used by UGCFv2 (320 channels) this is ~223k parameters instead of the
    ~9.6M a dense ``2C -> C`` stack of the same depth would cost.

    Its role is the fallback path of the fusion: when both phases are unreliable
    or they disagree, the network re-reads local neighbourhood structure from the
    raw concatenated features rather than trusting either phase's evidence.
    """

    def __init__(
        self,
        conv_op: Type[nn.Module],
        channels: int,
        kernel_size: Sequence[int],
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if conv_op is not nn.Conv3d:
            raise ValueError("VCAU-Net v2 only supports 3D convolutions.")
        kernel_size = validate_kernel_size(kernel_size)
        width = 2 * int(channels)
        padding = tuple(k // 2 for k in kernel_size)
        self.depthwise = conv_op(width, width, kernel_size, 1, padding, groups=width, bias=False)
        self.depthwise_norm = _norm(norm_op, width, norm_op_kwargs)
        self.project = ConvNormAct3D(
            conv_op, width, int(channels), (1, 1, 1), norm_op, norm_op_kwargs
        )

    def forward(self, f2: torch.Tensor, f5_aligned: torch.Tensor) -> torch.Tensor:
        if f2.shape != f5_aligned.shape:
            raise ValueError(
                f"Phase features must match, got {tuple(f2.shape)} vs "
                f"{tuple(f5_aligned.shape)}."
            )
        joined = torch.cat((f2, f5_aligned), dim=1)
        return self.project(torch.nn.functional.gelu(self.depthwise_norm(self.depthwise(joined))))
