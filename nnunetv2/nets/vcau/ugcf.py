"""Uncertainty-Disagreement Guided Cross-Phase Fusion (UGCFv2).

What was wrong with v1
----------------------
v1 fused with ``w2 = R2 / (R2 + R5)`` where ``R = exp(-U / tau)``. That ratio
depends only on the *difference* ``U5 - U2``, so it cannot distinguish

* "both phases are reliable here" (``U2 = U5 = 0.1``) from
* "neither phase is reliable here" (``U2 = U5 = 0.9``) --

both give ``w2 = w5 = 0.5``. v1 therefore degenerates to a plain average exactly
where it should have fallen back to something else, and the failure is invisible
in the loss. Worse, both auxiliary heads see the same shared encoder, the same
ground truth and the same loss, so ``p2 ~ p5`` and hence ``U2 ~ U5`` is a
plausible outcome, not a pathological one.

What v2 models instead
----------------------
Four quantities are made explicit and combined, rather than collapsed into one
ratio:

1. **Phase-specific uncertainty** ``U2``, ``U5`` -> relative selection
   ``alpha2``, ``alpha5``. Answers *which phase is more trustworthy here?*
   (Still a ratio, but now it is only used for the relative decision.)
2. **Joint uncertainty** ``U_joint = (U2 + U5)/2`` -> ``g_u = exp(-U_joint/tau_joint)``.
   Answers *is either phase trustworthy at all?* This is the term v1 lacked.
3. **Prediction disagreement** ``D_p = |p2 - p5|`` -> a learned gate ``g_d``.
   Answers *do the two phases agree about the lesion?* A disagreement is a
   different failure from an uncertainty: the phases can both be confident and
   still conflict.
4. **Feature-level DCE difference** ``DeltaF = |F2 - F5_aligned|``, kept as an
   explicit input so the fusion is never forced to average the enhancement
   difference away -- that difference is the diagnostic signal itself.

The joint confidence ``G = g_u * g_d`` then reads: *should this voxel trust the
uncertainty-weighted phase fusion, or fall back to local context?* The fusion is

    F_phase = alpha2*F2 + alpha5*F5_aligned
    F_ctx   = context([F2, F5_aligned])
    F_fuse  = phi([G*F_phase, (1-G)*F_ctx, DeltaF]) + F_phase

Gradient policy
---------------
Every quantity derived from the auxiliary predictions (``p``, ``U``, ``R``,
``alpha``, ``g_u``, ``D_p``, ``g_d``, ``G``) is **detached**. The auxiliary heads
are trained only by their own supervised term ``L_aux``. This is not a detail:
if gradients could flow through ``U`` into the fusion weights, the network could
raise its own fusion weight simply by becoming over-confident, a degenerate
solution that improves the weighting without improving the prediction. Detaching
makes the weights a *measurement* of reliability rather than an objective.

Feature tensors (``F2``, ``F5_aligned``) are of course *not* detached -- they
carry the segmentation gradient.
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional, Sequence, Type

import torch
from torch import nn

from nnunetv2.nets.vcau.ugcf import AuxSegHead3D  # reused unchanged
from nnunetv2.nets.vcau_v2.blocks import (
    ContextBranch3D,
    DepthwiseSeparableBlock3D,
    validate_kernel_size,
)

__all__ = [
    "AuxSegHead3D",
    "normalized_binary_entropy",
    "PhaseUncertainty",
    "DisagreementGate",
    "UGCFv2Output",
    "UGCFv2",
]


def normalized_binary_entropy(foreground_probability: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalised binary entropy of a foreground probability, in ``[0, 1]``.

    ``U(p) = -[p log(p + eps) + (1-p) log(1-p + eps)] / log 2``. Peaks at 1 for
    ``p = 0.5`` and is 0 for a fully confident prediction.

    Normalising by ``log 2`` (rather than using nats) makes the value directly
    comparable to the ``tau_joint`` threshold, so the gate is read in units of
    "fraction of maximal ambiguity".
    """
    p = foreground_probability
    if p.numel() and (p.min() < 0 or p.max() > 1):
        raise ValueError(
            f"Expected probabilities in [0, 1], got range [{float(p.min())}, {float(p.max())}]."
        )
    entropy = -(p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps))
    return (entropy / math.log(2.0)).clamp_(0.0, 1.0)


class PhaseUncertainty(nn.Module):
    """Foreground probability and normalised entropy of one phase's aux logits.

    Both outputs are **detached**: they are measurements consumed by the fusion
    gate and the alignment weight, never a route for the fusion objective to
    manipulate the auxiliary predictions.

    Shapes:
        logits ``[B, num_classes, D, H, W]``
        p      ``[B, 1, D, H, W]`` tumour probability
        U      ``[B, 1, D, H, W]`` normalised entropy in ``[0, 1]``
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor):
        if logits.ndim < 3:
            raise ValueError(
                f"Expected logits with a spatial dimension, got {tuple(logits.shape)}."
            )
        if logits.shape[1] < 2:
            raise ValueError(
                f"Expected nnU-Net style per-class logits (background + tumour), got "
                f"{logits.shape[1]} channel(s)."
            )
        probability = torch.softmax(logits.detach().float(), dim=1)[:, 1:2]
        return probability, normalized_binary_entropy(probability, self.eps)


class DisagreementGate(nn.Module):
    """Tiny learned gate over ``[U2, U5, D_p]`` -> ``g_d`` in ``(0, 1)``.

    ``D_p`` is fed in explicitly rather than being folded into a fixed formula
    such as ``g_d = 1 - D_p``. Two reasons: the explicit input keeps the gate
    interpretable (its response to disagreement can be plotted), and a learned
    response can express that some disagreement is benign -- adjacent phase
    differences at a lesion rim are expected -- while disagreement at a lesion
    core is not. A hard ``1 - D_p`` cannot.

    The parameter count is deliberately negligible (``3*h + h + h + 1``, i.e. 41
    at ``hidden=8``) so the gate cannot become a second fusion network in
    disguise.
    """

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}.")
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Conv3d(3, self.hidden, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv3d(self.hidden, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        # Zero-init the *final* projection only, so the gate starts neutral
        # (sigmoid(0) = 0.5, i.e. G = g_u / 2) and learns its response from there.
        #
        # Zero-initialising the first layer as well would freeze the gate for
        # good, not just for a step: with a zero first convolution its output is
        # zero, so the final layer's weight gradient is zero; and with a zero
        # final layer the first layer's weight gradient is zero too. Neither layer
        # could ever leave zero and g_d would stay pinned at 0.5 forever. Leaving
        # the first layer at its default init keeps its output non-zero, so the
        # final layer's gradient is non-zero immediately -- and once that layer
        # moves off zero, the first layer starts receiving gradient as well.
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, uncertainty_c2: torch.Tensor, uncertainty_c5: torch.Tensor, disagreement: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat((uncertainty_c2, uncertainty_c5, disagreement), dim=1)
        return self.net(gate_input.to(next(self.parameters()).dtype))


class UGCFv2Output(NamedTuple):
    """Result of one UGCFv2 step.

    Attributes:
        fused: ``[B, C, D, H, W]`` fused skip feature.
        reliability_c2 / reliability_c5: ``[B, 1, D, H, W]`` detached ``q2``/``q5``,
            consumed by 3D-SPCA's alignment weight.
        statistics: detached scalars for training diagnostics.
    """

    fused: torch.Tensor
    reliability_c2: torch.Tensor
    reliability_c5: torch.Tensor
    statistics: Dict[str, torch.Tensor]


class UGCFv2(nn.Module):
    """Per-voxel uncertainty-disagreement guided fusion of the two phase features.

    Args:
        conv_op: plans convolution class.
        channels: feature width ``C`` of both phases at this stage.
        kernel_size: 3-tuple of odd ints, from the plans.
        tau_phase: temperature of the *relative* phase selection. Only the ratio
            ``q2/q5 = exp(-(U2-U5)/tau_phase)`` matters, so this controls how
            sharply an entropy difference turns into a weight difference.
        tau_joint: temperature of the *joint* reliability gate. Unlike
            ``tau_phase`` this one is absolute: it sets how much ambiguity is
            tolerated before the phase path is abandoned for context.
        gate_hidden: width of the disagreement gate.
        use_joint_reliability / use_disagreement_gate: ablation switches. Setting
            one to False replaces its factor of ``G`` with 1, leaving everything
            else (including the auxiliary heads and ``F_phase``) identical, so the
            ablation changes exactly one factor.
        norm_op / norm_op_kwargs: plans normalisation.
    """

    def __init__(
        self,
        conv_op: Type[nn.Module],
        channels: int,
        kernel_size: Sequence[int] = (3, 3, 3),
        tau_phase: float = 0.5,
        tau_joint: float = 0.25,
        gate_hidden: int = 8,
        use_joint_reliability: bool = True,
        use_disagreement_gate: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if tau_phase <= 0 or tau_joint <= 0:
            raise ValueError(
                f"tau_phase and tau_joint must be strictly positive, got {tau_phase} "
                f"and {tau_joint}."
            )
        kernel_size = validate_kernel_size(kernel_size)
        self.channels = int(channels)
        self.tau_phase = float(tau_phase)
        self.tau_joint = float(tau_joint)
        self.eps = float(eps)
        self.use_joint_reliability = bool(use_joint_reliability)
        self.use_disagreement_gate = bool(use_disagreement_gate)

        self.uncertainty = PhaseUncertainty(eps=eps)
        self.disagreement_gate = DisagreementGate(gate_hidden) if self.use_disagreement_gate else None
        self.context = ContextBranch3D(
            conv_op, self.channels, kernel_size, norm_op, norm_op_kwargs
        )
        self.phi = DepthwiseSeparableBlock3D(
            conv_op, 3 * self.channels, self.channels, kernel_size, norm_op, norm_op_kwargs
        )
        # Zero-init the final projection so the operator starts as the identity on
        # F_phase; without it a random fusion would distort the skip pyramid that
        # the decoder expects to receive.
        last = self.phi.project.conv
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    # ------------------------------------------------------------------ internals
    def _weights(
        self, uncertainty_c2: torch.Tensor, uncertainty_c5: torch.Tensor
    ):
        """Relative phase weights, joint reliability, disagreement, gate value."""
        q2 = torch.exp(-uncertainty_c2 / self.tau_phase)
        q5 = torch.exp(-uncertainty_c5 / self.tau_phase)
        denominator = q2 + q5 + self.eps
        alpha2 = q2 / denominator
        alpha5 = q5 / denominator
        return q2, q5, alpha2, alpha5

    @staticmethod
    def _summary(name: str, value: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat = value.detach().float().reshape(-1)
        return {f"{name}_mean": flat.mean(), f"{name}_std": flat.std(unbiased=False)}

    # -------------------------------------------------------------------- forward
    def forward(
        self,
        f2: torch.Tensor,
        f5_aligned: torch.Tensor,
        logits_c2: torch.Tensor,
        logits_c5: torch.Tensor,
    ) -> UGCFv2Output:
        if f2.shape != f5_aligned.shape:
            raise ValueError(
                f"Phase features must match, got {tuple(f2.shape)} vs "
                f"{tuple(f5_aligned.shape)}."
            )
        for name, logits in (("c2", logits_c2), ("c5", logits_c5)):
            if tuple(logits.shape[2:]) != tuple(f2.shape[2:]):
                raise ValueError(
                    f"Auxiliary logits for {name} have spatial shape {tuple(logits.shape[2:])} "
                    f"but the features have {tuple(f2.shape[2:])}."
                )

        dtype = f2.dtype
        p2, u2 = self.uncertainty(logits_c2)
        p5, u5 = self.uncertainty(logits_c5)

        # 1. relative phase selection: which phase is more trustworthy here?
        q2, q5, alpha2, alpha5 = self._weights(u2, u5)
        f_phase = alpha2.to(dtype) * f2 + alpha5.to(dtype) * f5_aligned

        # 2. joint reliability: is either phase trustworthy at all?
        u_joint = 0.5 * (u2 + u5)
        if self.use_joint_reliability:
            g_u = torch.exp(-u_joint / self.tau_joint)
        else:
            g_u = torch.ones_like(u_joint)

        # 3. disagreement: do the phases agree about the lesion?
        disagreement = (p2 - p5).abs()
        if self.disagreement_gate is not None:
            g_d = self.disagreement_gate(u2, u5, disagreement)
        else:
            g_d = torch.ones_like(disagreement)

        # 4. joint confidence and the context fallback.
        G = (g_u * g_d).to(dtype)
        delta_f = (f2 - f5_aligned).abs()
        f_context = self.context(f2, f5_aligned)
        evidence = torch.cat((G * f_phase, (1.0 - G) * f_context, delta_f), dim=1)
        fused = self.phi(evidence) + f_phase

        statistics: Dict[str, torch.Tensor] = {}
        statistics.update(self._summary("alpha2", alpha2))
        statistics.update(self._summary("alpha5", alpha5))
        statistics.update(self._summary("U2", u2))
        statistics.update(self._summary("U5", u5))
        statistics.update(self._summary("D_p", disagreement))
        statistics.update(self._summary("G", g_u * g_d))
        # How far the phase selection departs from an uninformative 50/50 split.
        # This is the single most important number for "is UGCFv2 doing anything".
        statistics["alpha2_deviation_mean"] = (alpha2 - 0.5).abs().detach().float().mean()

        return UGCFv2Output(
            fused=fused,
            reliability_c2=q2,
            reliability_c5=q5,
            statistics=statistics,
        )
