"""Selective 3D Structural-Preserving Cross-phase Alignment (3D-SPCA v2).

The same idea as v1's ``SPCA3D`` -- remove the residual spatial mismatch between
the C2 and C5 acquisitions while preserving the genuine enhancement difference --
but applied at only two stage indices instead of all six, and with the
displacement bound configured per stage instead of derived from a formula.

Why the stage policy is selective
---------------------------------
* **Stage 0/1** (full and half resolution): the features there are dominated by
  grey level, edge and texture. C2 and C5 differ mostly by contrast uptake at
  those scales, and an alignment operator driven by such features can explain a
  real enhancement difference away as a displacement. The residual
  misregistration is also already small relative to the voxel size there.
* **Stage 5** (the bottleneck): at ``1x2x2`` there is no meaningful spatial
  extent for a displacement field to act on.

Stage 2 and 3 are the scales where the residual mismatch is comparable to the
voxel size and the features are structural rather than textural.

Warp contract
-------------
``Warp3D`` is imported from v1 unchanged. It encodes the empirically calibrated
``grid_sample`` rules: grid last axis is ``(x, y, z) = (W, H, D)``, ``mode`` must
be ``'bilinear'``, and with ``align_corners=False`` the identity grid is
``(2i + 1) / N - 1`` rather than zero. Re-deriving it here would duplicate a
contract that is already pinned down by tests, so it is reused, not rewritten.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Sequence, Type

import torch
from torch import nn

from nnunetv2.nets.vcau.spca import Warp3D, build_identity_grid  # reused, not modified
from nnunetv2.nets.vcau_v2.blocks import ConvNormAct3D, validate_kernel_size

__all__ = [
    "Warp3D",
    "build_identity_grid",
    "SPCA_MAX_OFFSETS",
    "SPCAv2",
    "SPCAv2Output",
    "BoundedFlowPredictor3D",
    "flow_statistics",
    "resolve_max_offsets",
]

#: Per-stage displacement bound in voxels at *that stage's* resolution.
#
#: Centralised on purpose: the bound is a method hyper-parameter (it encodes how
#: much residual misregistration the registration pipeline is assumed to leave
#: behind), so it must not be scattered as literals near the operator.
#: Stage 2 spans 8x16x16 for a 64x160x160 patch and stage 3 spans 4x8x8; one
#: voxel at stage 3 already covers two voxels at stage 2 and eight at stage 0, so
#: the bound shrinks with depth.
SPCA_MAX_OFFSETS: Dict[int, int] = {2: 2, 3: 1}


class SPCAv2Output(NamedTuple):
    """Result of one 3D-SPCA v2 step.

    Attributes:
        f5_aligned: ``[B, C, D, H, W]`` C5 features warped into the C2 frame.
        z2: ``[B, C, D, H, W]`` structural projection of the reference (C2).
        z5_aligned: ``[B, C, D, H, W]`` warped structural projection of C5.
        flow: ``[B, 3, D, H, W]`` bounded displacement, voxel units, ``(d, h, w)``.
    """

    f5_aligned: torch.Tensor
    z2: torch.Tensor
    z5_aligned: torch.Tensor
    flow: torch.Tensor


def flow_statistics(flow: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Detached scalar summary of a displacement field, for training diagnostics.

    ``mean``/``std`` answer "is the warp doing anything at all", ``p95``/``max``
    answer "is it saturating the bound". A ``mean`` that decays to zero over
    training means 3D-SPCA has collapsed to the identity mapping.
    """
    values = flow.detach().float().reshape(-1)
    return {
        "mean": values.mean(),
        "std": values.std(unbiased=False),
        "p95": torch.quantile(values, 0.95),
        "max": values.abs().amax(),
    }


class BoundedFlowPredictor3D(nn.Module):
    """Bounded residual displacement field between two structural volumes.

    ``O = s_l * tanh(FlowNet([Z2, Z5]))``. The ``tanh`` saturation is the hard
    bound: no batch, however badly conditioned, can produce a displacement larger
    than ``max_offset`` voxels along any axis.

    The head is zero-initialised so training starts from the identity warp. That
    is deliberate -- a randomly initialised deformation would corrupt the C5
    features before the network has learned anything -- but it also means the
    diagnostics must watch for the opposite failure, a warp that *stays* at the
    identity (see :func:`flow_statistics` and the trainer's collapse warning).

    Args:
        channels: feature width at the current scale.
        max_offset: per-axis bound in voxels, ``(d, h, w)``.
        kernel_size: 3-tuple of odd ints, from the plans.
    """

    def __init__(
        self,
        conv_op: Type[nn.Module],
        channels: int,
        max_offset: int,
        kernel_size: Sequence[int] = (3, 3, 3),
        norm_op: Type[nn.Module] | None = None,
        norm_op_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        bound = float(max_offset)
        if bound <= 0:
            raise ValueError(f"max_offset must be positive, got {max_offset}.")
        kernel_size = validate_kernel_size(kernel_size)

        self.register_buffer(
            "max_offset", torch.tensor([bound, bound, bound]).view(1, 3, 1, 1, 1)
        )
        self.stem = ConvNormAct3D(
            conv_op, 2 * int(channels), int(channels), (1, 1, 1), norm_op, norm_op_kwargs
        )
        self.body = ConvNormAct3D(
            conv_op, int(channels), int(channels), kernel_size, norm_op, norm_op_kwargs
        )
        self.head = conv_op(int(channels), 3, 1, 1, 0, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z2: torch.Tensor, z5: torch.Tensor) -> torch.Tensor:
        if z2.shape != z5.shape:
            raise ValueError(
                f"Structure features must match, got {tuple(z2.shape)} vs {tuple(z5.shape)}."
            )
        x = self.body(self.stem(torch.cat((z2, z5), dim=1)))
        return torch.tanh(self.head(x)) * self.max_offset.to(x.dtype)


class SPCAv2(nn.Module):
    """Structure-preserving cross-phase alignment at a single scale.

    A single ``psi`` is shared by both phases: it defines one phase-agnostic
    structural subspace, so the alignment cannot be satisfied by re-coding one
    phase's contrast into a different basis.

    Shapes:
        f2, f5 ``[B, C, D, H, W]``
        flow   ``[B, 3, D, H, W]``
    """

    def __init__(
        self,
        conv_op: Type[nn.Module],
        channels: int,
        max_offset: int,
        kernel_size: Sequence[int] = (3, 3, 3),
        norm_op: Type[nn.Module] | None = None,
        norm_op_kwargs: dict | None = None,
        padding_mode: str = "border",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.psi = ConvNormAct3D(
            conv_op, int(channels), int(channels), (1, 1, 1), norm_op, norm_op_kwargs
        )
        self.flow_predictor = BoundedFlowPredictor3D(
            conv_op, int(channels), max_offset, kernel_size, norm_op, norm_op_kwargs
        )
        self.warp = Warp3D(padding_mode=padding_mode)

    def forward(self, f2: torch.Tensor, f5: torch.Tensor) -> SPCAv2Output:
        if f2.shape != f5.shape:
            raise ValueError(
                f"Phase features must match, got {tuple(f2.shape)} vs {tuple(f5.shape)}."
            )
        z2 = self.psi(f2)
        z5 = self.psi(f5)
        flow = self.flow_predictor(z2, z5)
        return SPCAv2Output(
            f5_aligned=self.warp(f5, flow),
            z2=z2,
            z5_aligned=self.warp(z5, flow),
            flow=flow,
        )


def resolve_max_offsets(
    stages: Sequence[int], table: Dict[int, int] | None = None
) -> Dict[int, int]:
    """Look up the bound for each SPCA stage, failing fast on a missing entry.

    A missing entry means a stage was added to the SPCA policy without deciding
    how far it may displace, which would silently reintroduce a magic number.
    """
    table = SPCA_MAX_OFFSETS if table is None else table
    resolved: Dict[int, int] = {}
    for stage in stages:
        if stage not in table:
            raise ValueError(
                f"No max_offset configured for SPCA stage {stage}. Add it to "
                f"SPCA_MAX_OFFSETS ({sorted(table)}) rather than hardcoding it at the call site."
            )
        resolved[int(stage)] = int(table[stage])
    return resolved
