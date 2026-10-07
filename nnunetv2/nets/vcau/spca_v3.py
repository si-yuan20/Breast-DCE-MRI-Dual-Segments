"""Symmetric 3D Structural-Preserving Cross-phase Alignment (3D-SPCA v3).

The method, in one line
-----------------------
**Symmetric correspondence estimation + asymmetric reference alignment.**

Correspondence between the C2 and C5 acquisitions is estimated in *both*
directions by one shared predictor; only the ``C5 -> C2`` half of that
correspondence is ever applied to features. C2 stays the fixed reference, so the
fused skips, the decoder and the exported segmentation all live in the C2 frame
-- exactly as in v2. What changes is how well-posed the flow estimation is.

Why v2's estimator was under-constrained
----------------------------------------
Audit findings, each of which this module addresses (see
``docs/superpowers/specs/2026-09-21-spca-v3-symmetric-design.md``):

1. **No explicit correspondence cues.** v2 fed ``cat(Z2, Z5)`` into a linear
   ``1x1x1`` stem. A linear map can form ``Z2 - Z5`` but neither ``|Z2 - Z5|``
   (nonlinear) nor ``Z2 * Z5`` (bilinear). The mismatch-magnitude and
   common-activation signals were therefore not representable at all. v3 builds
   them explicitly in :func:`correspondence_descriptor`.
2. **Directional bias.** One flow, no reverse estimate, no cycle constraint; the
   stem could key on the *slot position* of Z2 vs Z5 instead of on structure.
3. **``L_align`` alone cannot pin the flow down.** psi is shared *and trainable*,
   so driving ``1 - cos`` to zero by making psi contrast-suppressing is a
   degenerate solution that also sends the flow to zero. Cosine similarity is
   additionally flat for small displacements, i.e. exactly in the 1-2 voxel
   regime this method targets.
4. **Regularisation cannot resist identity collapse.** ``TV(O) = 0`` for any
   *constant* field, including ``O = 0``: the smoothness prior's global minimum
   *is* the collapsed solution. Folding is unconstrained (TV bounds first
   differences, not ``det J``). Direction consistency is undefined with a single
   field.
5. **The diagnostics were wrong.** v2 reported quantiles of *signed* components,
   so a healthy, roughly symmetric warp yields ``mean(signed) ~ 0`` and the
   trainer's collapse detector fired spuriously. :func:`flow_statistics` reports
   component magnitudes and per-voxel displacement magnitudes separately.

What is honestly *not* claimed
------------------------------
Neither ``L_smooth`` nor ``L_inv`` resists identity collapse: both are exactly
zero at ``O = 0``. The only force that ever moves the warp is ``L_align``.
Symmetric estimation improves *identifiability and symmetry* of the flow; it does
not manufacture the drive to move. That is why the collapse diagnostics in
:func:`flow_statistics` remain the instrument of record for "is the module doing
anything".

Coordinate contract (inherited from ``Warp3D``, verified not assumed)
--------------------------------------------------------------------
The flow is ``[B, 3, D, H, W]`` in **voxel units** with channel order
``(d, h, w)`` -- channel 0 is the **D** axis, channel 1 the H axis, channel 2 the
W axis. That is ``Warp3D``'s live contract: it scales by ``[2/D, 2/H, 2/W]`` and
``flip(1)``-s into the sampling grid's ``(x, y, z) = (W, H, D)`` last axis. An
earlier draft of this method's specification stated ``(x, y, z) = (W, H, D)`` for
the channel order; that is the *grid* order, not the flow order, and acting on it
would ship a silently transposed displacement field. There is an axis-by-axis
direction test in ``test/test_vcau_v3.py`` to keep this pinned.

Warping follows the usual backward convention ``out[i] = in[i + flow[i]]``, so a
positive displacement moves image content towards lower indices.

Inverse consistency reuses ``Warp3D`` -- no unit conversion
-----------------------------------------------------------
A displacement field is itself a ``[B, 3, D, H, W]`` tensor, so
``Warp3D(O_2to5, O_5to2)`` is shape-valid and semantically exact::

    Warp3D(O_2to5, O_5to2)[x] = O_2to5[x + O_5to2(x)]

and inverse consistency is ``O_5to2(x) + O_2to5(x + O_5to2(x)) ~ 0``, which is
precisely what :func:`inverse_consistency_loss` measures. Both fields are
voxel-unit and in the same channel order, so **no conversion is applied**. This is
proven by a translate-then-invert test rather than asserted.

Note that ``L_inv`` has zero gradient at exact initialisation (both flows are 0
and ``|x|`` has a zero subgradient at 0). It becomes active once ``L_align`` has
moved the forward flow; it cannot by itself escape the collapsed state.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Sequence, Type

import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.nets.vcau.spca import Warp3D, build_identity_grid  # reused, not modified
from nnunetv2.nets.vcau_v2.blocks import ConvNormAct3D, validate_kernel_size
from nnunetv2.nets.vcau_v2.spca import (  # single source of truth for the policy
    SPCA_MAX_OFFSETS,
    resolve_max_offsets,
)

__all__ = [
    "Warp3D",
    "build_identity_grid",
    "SPCA_MAX_OFFSETS",
    "resolve_max_offsets",
    "DESCRIPTOR_MODES",
    "SPCAv3Output",
    "StructuralAdapter3D",
    "SymmetricFlowPredictor3D",
    "SPCAv3",
    "correspondence_descriptor",
    "warp_vector_field",
    "inverse_consistency_loss",
    "flow_statistics",
]

#: ``symmetric`` -- the 4C descriptor ``cat(a, b, |a-b|, a*b)``.
#: ``concat``    -- the 2C descriptor ``cat(a, b)``, i.e. v2's input, kept so the
#: descriptor itself can be ablated without touching anything else.
DESCRIPTOR_MODES = ("symmetric", "concat")


def correspondence_descriptor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Explicit correspondence cues between two structural volumes.

    ``C(a, b) = cat(a, b, |a - b|, a * b)`` on the channel axis, giving ``4C``.

    The two appended blocks are the point of the module:

    * ``|a - b|`` is the **structural mismatch**. A displacement field has to be
      driven by how far apart the phases are, and that quantity is nonlinear --
      a ``1x1x1`` stem over ``cat(a, b)`` cannot form it.
    * ``a * b`` is the **common activation**: where both phases agree. It is
      bilinear, so a linear stem cannot form it either.

    Args:
        a, b: ``[B, C, D, H, W]`` structure features. Order matters: ``C(a, b)``
            drives the ``a -> b`` direction.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Correspondence needs matching features, got {tuple(a.shape)} vs {tuple(b.shape)}."
        )
    return torch.cat((a, b, (a - b).abs(), a * b), dim=1)


class StructuralAdapter3D(nn.Module):
    """Light structural adapter, shared by both phases.

    ``Z = F_proj + LocalStructuralRefinement(F_proj)`` with

        F_proj = Conv1x1x1 -> Norm -> GELU
        LocalStructuralRefinement = DepthwiseConv(k^3) -> Norm -> GELU

    The refinement is depthwise so the local mixing costs ``k^3 * C`` rather than
    ``k^3 * C^2``; the residual is licensed because the channel width is
    unchanged, and it means the adapter has to learn a *correction* to the point
    projection rather than reproduce it. One instance is used for both phases --
    there is deliberately no ``psi_c2``/``psi_c5`` pair, because a phase-specific
    pair would let the alignment be satisfied by re-coding one phase's contrast
    into a different basis instead of by moving anything.

    Args:
        conv_op: plans convolution class (must be 3D).
        channels: feature width ``C`` at this scale.
        kernel_size: 3-tuple of odd ints, from the plans.
        norm_op / norm_op_kwargs: plans normalisation.
    """

    def __init__(
        self,
        conv_op: Type[nn.Module],
        channels: int,
        kernel_size: Sequence[int] = (3, 3, 3),
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if conv_op is not nn.Conv3d:
            raise ValueError(
                f"3D-SPCA v3 only supports 3D convolutions, got "
                f"{getattr(conv_op, '__name__', conv_op)}."
            )
        channels = int(channels)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        kernel_size = validate_kernel_size(kernel_size)

        self.channels = channels
        self.project = ConvNormAct3D(
            conv_op, channels, channels, (1, 1, 1), norm_op, norm_op_kwargs
        )
        padding = tuple(k // 2 for k in kernel_size)
        self.depthwise = conv_op(
            channels, channels, kernel_size, 1, padding, groups=channels, bias=False
        )
        self.depthwise_norm = (
            norm_op(channels, **(norm_op_kwargs or {})) if norm_op is not None else nn.Identity()
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        projected = self.project(f)
        local = F.gelu(self.depthwise_norm(self.depthwise(projected)))
        return projected + local


class SymmetricFlowPredictor3D(nn.Module):
    """Bounded displacement field from a correspondence descriptor.

    ``O = s_l * tanh(Phi(C))``. The ``tanh`` saturation is the hard bound: no
    batch, however badly conditioned, can produce a displacement larger than
    ``max_offset`` voxels along any axis.

    **One instance serves both directions.** ``forward`` is called with
    ``C(Z2, Z5)`` and with ``C(Z5, Z2)``; parameter sharing is what forces a
    single explanation of the correspondence rather than two independent
    one-sided ones, and it is asserted by an identity test.

    The head is zero-initialised so training starts from the identity warp -- a
    randomly initialised deformation would corrupt the C5 features before the
    network has learned anything. Only the *final* layer is zeroed: zeroing the
    stem as well would starve the earlier layers of gradient permanently, not
    just for one step.

    Args:
        conv_op: plans convolution class.
        channels: feature width ``C`` at the current scale.
        max_offset: per-axis bound in voxels.
        descriptor_channels: width of the descriptor this predictor consumes
            (``4C`` for the symmetric descriptor, ``2C`` for ``concat``).
        kernel_size: 3-tuple of odd ints, from the plans.
        norm_op / norm_op_kwargs: plans normalisation.
    """

    def __init__(
        self,
        conv_op: Type[nn.Module],
        channels: int,
        max_offset: int,
        descriptor_channels: int,
        kernel_size: Sequence[int] = (3, 3, 3),
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
    ) -> None:
        super().__init__()
        bound = float(max_offset)
        if bound <= 0:
            raise ValueError(f"max_offset must be positive, got {max_offset}.")
        channels = int(channels)
        descriptor_channels = int(descriptor_channels)
        if descriptor_channels <= 0:
            raise ValueError(f"descriptor_channels must be positive, got {descriptor_channels}.")
        kernel_size = validate_kernel_size(kernel_size)

        self.channels = channels
        self.descriptor_channels = descriptor_channels
        self.register_buffer(
            "max_offset", torch.tensor([bound, bound, bound]).view(1, 3, 1, 1, 1)
        )
        self.stem = ConvNormAct3D(
            conv_op, descriptor_channels, channels, (1, 1, 1), norm_op, norm_op_kwargs
        )
        self.body = ConvNormAct3D(
            conv_op, channels, channels, kernel_size, norm_op, norm_op_kwargs
        )
        self.head = conv_op(channels, 3, 1, 1, 0, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.shape[1] != self.descriptor_channels:
            raise ValueError(
                f"Expected a {self.descriptor_channels}-channel correspondence descriptor, got "
                f"{descriptor.shape[1]}."
            )
        x = self.body(self.stem(descriptor))
        return torch.tanh(self.head(x)) * self.max_offset.to(x.dtype)


class SPCAv3Output(NamedTuple):
    """Result of one 3D-SPCA v3 step.

    Attributes:
        f5_aligned: ``[B, C, D, H, W]`` C5 features warped into the C2 frame. The
            only field the downstream fusion ever sees.
        z2: ``[B, C, D, H, W]`` structural projection of the reference (C2).
        z5: ``[B, C, D, H, W]`` structural projection of C5, *unaligned*.
        z5_aligned: ``[B, C, D, H, W]`` warped structural projection of C5.
        flow_5_to_2: ``[B, 3, D, H, W]`` bounded C5 -> C2 displacement, voxels.
        flow_2_to_5: ``[B, 3, D, H, W]`` bounded C2 -> C5 displacement, voxels, or
            ``None`` for the one-way ablation arm. Never applied to features; it
            exists only to regularise the correspondence.
    """

    f5_aligned: torch.Tensor
    z2: torch.Tensor
    z5: torch.Tensor
    z5_aligned: torch.Tensor
    flow_5_to_2: torch.Tensor
    flow_2_to_5: Optional[torch.Tensor]


def warp_vector_field(
    displacement: torch.Tensor, flow: torch.Tensor, warp: Warp3D
) -> torch.Tensor:
    """Resample a displacement field with another displacement field.

    ``Warp3D`` is reused verbatim. Its contract applies unchanged because a
    displacement field is an ordinary ``[B, 3, D, H, W]`` tensor: 3 channels, same
    spatial shape, same batch. **No unit conversion is performed and none is
    needed** -- both fields are voxel-unit with channel order ``(d, h, w)``, which
    is what ``Warp3D`` expects of both its arguments.

    Args:
        displacement: ``[B, 3, ...]`` field to resample, voxels.
        flow: ``[B, 3, ...]`` sampling field, voxels.
        warp: the caller's ``Warp3D`` instance, shared rather than re-created so
            the padding mode stays in one place.
    """
    if displacement.shape[1] != 3:
        raise ValueError(
            f"warp_vector_field resamples a displacement field, which needs 3 channels, got "
            f"{displacement.shape[1]}."
        )
    return warp(displacement, flow)


def inverse_consistency_loss(
    flow_5_to_2: torch.Tensor, flow_2_to_5: torch.Tensor, warp: Warp3D
) -> torch.Tensor:
    """``mean |O_5to2 + Warp(O_2to5, O_5to2)|``.

    After mapping ``5 -> 2``, the reverse field sampled at the corresponding
    location should point back and cancel the forward one. Where it does not, the
    two directions disagree about the same correspondence, and at least one of
    them is wrong.

    This is a *training-time* constraint on the correspondence estimate. The
    reverse field it consumes is never applied to features: C2 stays the fixed
    reference.

    Args:
        flow_5_to_2: ``[B, 3, ...]`` forward displacement, voxels.
        flow_2_to_5: ``[B, 3, ...]`` reverse displacement, voxels.
        warp: the caller's ``Warp3D`` instance.
    """
    if flow_5_to_2.shape != flow_2_to_5.shape:
        raise ValueError(
            f"Inverse consistency needs matching fields, got {tuple(flow_5_to_2.shape)} vs "
            f"{tuple(flow_2_to_5.shape)}."
        )
    backward = warp_vector_field(flow_2_to_5, flow_5_to_2, warp)
    return (flow_5_to_2 + backward).abs().mean()


def flow_statistics(flow: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Detached summary of a displacement field, for training diagnostics.

    Two families, because they answer different questions:

    * ``component_abs_*`` -- per-axis signed displacement magnitudes. ``max``
      answers "is it saturating the ``max_offset`` bound".
    * ``*_magnitude`` -- the per-voxel displacement length
      ``sqrt(dx^2 + dy^2 + dz^2)``. This is the family to watch for collapse: a
      healthy warp is roughly symmetric about zero, so the mean of *signed*
      components is near zero even when the field is doing a lot of work. v2
      reported signed-component quantiles under a ``magnitude`` name, which is
      why its collapse detector could fire on a perfectly active warp.

    Args:
        flow: ``[B, 3, D, H, W]`` displacement in voxels.
    """
    if flow.ndim != 5:
        raise ValueError(f"Expected a 5D [B, 3, D, H, W] flow, got {tuple(flow.shape)}.")
    if flow.shape[1] != 3:
        raise ValueError(f"Expected a 3-channel displacement field, got {flow.shape[1]}.")

    values = flow.detach().float()
    component = values.abs().reshape(-1)
    magnitude = values.square().sum(dim=1).sqrt().reshape(-1)
    return {
        "component_abs_mean": component.mean(),
        "component_abs_p95": torch.quantile(component, 0.95),
        "component_abs_max": component.amax(),
        "mean_magnitude": magnitude.mean(),
        "p95_magnitude": torch.quantile(magnitude, 0.95),
        "max_magnitude": magnitude.amax(),
    }


class SPCAv3(nn.Module):
    """Symmetric structure-preserving cross-phase alignment at a single scale.

    Pipeline::

        Z2 = psi(F2)                    Z5 = psi(F5)
        C_25 = C(Z2, Z5)                C_52 = C(Z5, Z2)
        O_5to2 = Phi(C_25)              O_2to5 = Phi(C_52)     <- one Phi, two calls
        F5_aligned = Warp(F5, O_5to2)   Z5_aligned = Warp(Z5, O_5to2)
        C2 is never warped.

    Both ``psi`` and ``Phi`` are single instances used for both phases and both
    directions respectively. That is the whole method: the correspondence is
    estimated symmetrically, and only one half of it is applied.

    Args:
        conv_op: plans convolution class.
        channels: feature width ``C`` at this scale.
        max_offset: per-axis displacement bound in voxels at *this* scale.
        kernel_size: 3-tuple of odd ints, from the plans.
        norm_op / norm_op_kwargs: plans normalisation.
        padding_mode: ``grid_sample`` padding for both warps.
        descriptor_mode: ``symmetric`` (4C, the method) or ``concat`` (2C, v2's
            input). The latter exists so the descriptor can be ablated in
            isolation from everything else.
        bidirectional: predict ``O_2to5`` as well. ``False`` reproduces the v2
            topology with v3's descriptor and adapter, which is the
            "descriptor only" ablation arm.

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
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        padding_mode: str = "border",
        descriptor_mode: str = "symmetric",
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if descriptor_mode not in DESCRIPTOR_MODES:
            raise ValueError(
                f"descriptor_mode must be one of {DESCRIPTOR_MODES}, got {descriptor_mode!r}."
            )
        channels = int(channels)
        self.channels = channels
        self.descriptor_mode = descriptor_mode
        self.bidirectional = bool(bidirectional)
        # 4C for the symmetric descriptor, 2C for the plain concatenation.
        self.descriptor_channels = (4 if descriptor_mode == "symmetric" else 2) * channels

        self.psi = StructuralAdapter3D(conv_op, channels, kernel_size, norm_op, norm_op_kwargs)
        self.flow_predictor = SymmetricFlowPredictor3D(
            conv_op, channels, max_offset, self.descriptor_channels, kernel_size,
            norm_op, norm_op_kwargs,
        )
        self.warp = Warp3D(padding_mode=padding_mode)

    def descriptor(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Correspondence descriptor for the ``a -> b`` direction."""
        if self.descriptor_mode == "symmetric":
            return correspondence_descriptor(a, b)
        return torch.cat((a, b), dim=1)

    def forward(self, f2: torch.Tensor, f5: torch.Tensor) -> SPCAv3Output:
        if f2.shape != f5.shape:
            raise ValueError(
                f"Phase features must match, got {tuple(f2.shape)} vs {tuple(f5.shape)}."
            )
        z2 = self.psi(f2)
        z5 = self.psi(f5)

        flow_5_to_2 = self.flow_predictor(self.descriptor(z2, z5))
        flow_2_to_5 = (
            self.flow_predictor(self.descriptor(z5, z2)) if self.bidirectional else None
        )

        return SPCAv3Output(
            f5_aligned=self.warp(f5, flow_5_to_2),
            z2=z2,
            z5=z5,
            z5_aligned=self.warp(z5, flow_5_to_2),
            flow_5_to_2=flow_5_to_2,
            flow_2_to_5=flow_2_to_5,
        )
