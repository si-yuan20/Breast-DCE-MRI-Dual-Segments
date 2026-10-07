"""Loss for VCAU-Net v2.

    L = L_main
      + lambda_aux      * L_aux        supervised auxiliary phase heads
      + lambda_align    * L_align      structure-preserving alignment (stages 2, 3)
      + lambda_smooth   * L_smooth     displacement-field TV (stages 2, 3)
      + lambda_boundary * L_boundary   distance-weighted boundary term (warm-up)

``L_main`` is not reimplemented. The trainer passes in
``super()._build_loss()`` -- the stock ``DeepSupervisionWrapper`` around
``DC_and_CE_loss`` -- so the segmentation objective is bit-for-bit the one the
nnU-Net baseline optimises and a performance difference cannot come from a
re-implemented Dice.

Boundary warm-up
----------------
v1 applied the level-set boundary term from a constant weight on step 0. The
level-set formulation (Kervadec et al.) deliberately ramps its weight from zero
because the signed-distance term has a very different scale from a region loss
and dominates early gradients, before the prediction has any plausible shape.
v2 ramps linearly from 0 to ``lambda_boundary`` over ``boundary_warmup_epochs``.
Whether the term earns its place at all is decided by the ``NoBoundary``
ablation arm, not by assumption -- loss stacking that does not pay for itself is
a liability in review.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from nnunetv2.nets.vcau.losses import boundary_distance_loss  # reused unchanged

__all__ = ["boundary_distance_loss", "VCAUNetV2Loss"]


class VCAUNetV2Loss(nn.Module):
    """Sums VCAU-Net v2's loss terms into one scalar.

    Args:
        main_loss: deep-supervision-wrapped nnU-Net loss from the parent trainer.
        aux_loss: per-scale loss for the auxiliary heads (Dice + CE, not DS-wrapped).
        num_classes: logit channels of the network output.
        has_regions: region-based training flag; unsupported here.
        ignore_label: forwarded to the boundary term, which rejects it.
        lambda_aux / lambda_align / lambda_smooth / lambda_boundary: weights.
        boundary_warmup_epochs: linear ramp length for the boundary weight. 0
            disables the ramp (constant weight from the first step).
        aux_scale_decay: per-scale decay of the auxiliary weight, mirroring
            nnU-Net's deep-supervision decay; normalised so ``lambda_aux`` stays
            comparable across stage counts.
        distance_radius: truncation radius of the boundary distance map.
    """

    def __init__(
        self,
        main_loss: nn.Module,
        aux_loss: nn.Module,
        num_classes: int,
        has_regions: bool,
        ignore_label: Optional[int],
        lambda_aux: float = 0.2,
        lambda_align: float = 0.1,
        lambda_smooth: float = 0.01,
        lambda_boundary: float = 0.05,
        boundary_warmup_epochs: int = 20,
        aux_scale_decay: float = 0.5,
        distance_radius: int = 6,
    ) -> None:
        super().__init__()
        if has_regions:
            raise ValueError(
                "VCAU-Net v2 implements the two-class breast tumour formulation; region-based "
                "training is not supported."
            )
        for name, value in (
            ("lambda_aux", lambda_aux),
            ("lambda_align", lambda_align),
            ("lambda_smooth", lambda_smooth),
            ("lambda_boundary", lambda_boundary),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if boundary_warmup_epochs < 0:
            raise ValueError(
                f"boundary_warmup_epochs must be non-negative, got {boundary_warmup_epochs}."
            )

        self.main_loss = main_loss
        self.aux_loss = aux_loss
        self.num_classes = int(num_classes)
        self.ignore_label = ignore_label
        self.lambda_aux = float(lambda_aux)
        self.lambda_align = float(lambda_align)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_boundary = float(lambda_boundary)
        self.boundary_warmup_epochs = int(boundary_warmup_epochs)
        self.aux_scale_decay = float(aux_scale_decay)
        self.distance_radius = int(distance_radius)
        self.current_epoch = 0

    # ------------------------------------------------------------------ schedule
    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def boundary_weight(self) -> float:
        """Current boundary weight, linearly ramped over the warm-up window."""
        if self.boundary_warmup_epochs <= 0:
            return self.lambda_boundary
        progress = min(1.0, max(0.0, self.current_epoch / float(self.boundary_warmup_epochs)))
        return self.lambda_boundary * progress

    # -------------------------------------------------------------------- helpers
    def _aux_target(self, full_resolution_target: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        """Nearest-neighbour resample the label map to an auxiliary head's grid.

        Derived from the full-resolution ground truth rather than indexing the
        deep-supervision target list: nnU-Net rounds DS targets with ``round()``
        while strided convolutions round with ``ceil``, so the two can disagree by
        one voxel on odd-sized axes.
        """
        resampled = torch.nn.functional.interpolate(
            full_resolution_target.float(), size=tuple(shape), mode="nearest"
        )
        return resampled.long()

    def _auxiliary_term(
        self, aux_c2: List[torch.Tensor], aux_c5: List[torch.Tensor], target: torch.Tensor
    ) -> torch.Tensor:
        if len(aux_c2) != len(aux_c5):
            raise ValueError(
                f"C2 and C5 must produce the same number of auxiliary heads, got "
                f"{len(aux_c2)} and {len(aux_c5)}."
            )
        weights = torch.tensor(
            [self.aux_scale_decay ** i for i in range(len(aux_c2))], dtype=torch.float32
        )
        weights = weights / weights.sum()
        total = torch.zeros((), device=aux_c2[0].device, dtype=torch.float32)
        for index, (logit_c2, logit_c5) in enumerate(zip(aux_c2, aux_c5)):
            resampled = self._aux_target(target, logit_c2.shape[2:])
            scale_weight = weights[index].to(logit_c2.device)
            for logit in (logit_c2, logit_c5):
                total = total + scale_weight * self.aux_loss(logit.float(), resampled)
        return total

    # -------------------------------------------------------------------- forward
    def forward(
        self,
        output: Union[dict, list, torch.Tensor],
        target,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        # Plain inference-style output -> stock nnU-Net loss only.
        if not isinstance(output, dict):
            main_only = self.main_loss(output, target)
            if return_components:
                return main_only, {"main": main_only}
            return main_only

        segmentation = output["seg"]
        main_term = self.main_loss(segmentation, target)
        components = {"main": main_term}
        total = main_term

        full_resolution_target = target[0] if isinstance(target, (list, tuple)) else target
        main_logits = segmentation[0] if isinstance(segmentation, (list, tuple)) else segmentation

        aux_c2 = output.get("aux_c2") or []
        aux_c5 = output.get("aux_c5") or []
        if self.lambda_aux > 0 and aux_c2 and aux_c5:
            components["aux"] = self._auxiliary_term(aux_c2, aux_c5, full_resolution_target)
            total = total + self.lambda_aux * components["aux"]

        boundary_weight = self.boundary_weight()
        if boundary_weight > 0:
            components["boundary"] = boundary_distance_loss(
                main_logits,
                full_resolution_target,
                self.num_classes,
                ignore_label=self.ignore_label,
                distance_radius=self.distance_radius,
            )
            total = total + boundary_weight * components["boundary"]

        align_terms = output.get("align") or []
        if self.lambda_align > 0 and align_terms:
            components["align"] = torch.stack([t.float() for t in align_terms]).mean()
            total = total + self.lambda_align * components["align"]

        smooth_terms = output.get("smooth") or []
        if self.lambda_smooth > 0 and smooth_terms:
            components["smooth"] = torch.stack([t.float() for t in smooth_terms]).mean()
            total = total + self.lambda_smooth * components["smooth"]

        if return_components:
            return total, components
        return total
