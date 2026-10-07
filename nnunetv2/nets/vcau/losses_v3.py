"""Loss for 3D-SPCA v3 / VCAU-Net v3.

    L = L_main
      + lambda_aux      * L_aux
      + lambda_align    * L_align      structure-preserving alignment   (stages 2, 3)
      + lambda_smooth   * L_smooth     bilateral displacement TV        (stages 2, 3)
      + lambda_inv      * L_inv        inverse consistency              (stages 2, 3)
      + lambda_boundary * L_boundary   distance-weighted boundary term  (warm-up)

This is :class:`~nnunetv2.nets.vcau_v2.losses.VCAUNetV2Loss` plus exactly one
term. Everything else -- the stock nnU-Net main loss, the auxiliary phase heads,
the boundary warm-up schedule -- is inherited unchanged, so a v2-vs-v3
comparison moves one variable.

Per-stage aggregation, and why it matters for the weights
---------------------------------------------------------
``L_smooth`` is ``TV(O_5to2) + TV(O_2to5)`` **summed inside one stage term**, and
the loss averages those stage terms. The two TVs are *not* appended as separate
entries: doing so would divide the per-stage smoothness by two and silently
halve ``lambda_smooth`` relative to v2, which would make the two arms' identical
hyper-parameter mean different things. Same reasoning for ``L_inv`` and
``L_align``: one term per stage, averaged across stages.

What is deliberately absent
---------------------------
No cycle, Jacobian-determinant, contrastive, attention or mutual-information
term. Each of those would have to earn its place against an ablation arm, and
stacking unproven regularisers is a liability in review rather than a
contribution.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from nnunetv2.nets.vcau_v2.losses import VCAUNetV2Loss

__all__ = ["VCAUNetV3Loss"]


class VCAUNetV3Loss(VCAUNetV2Loss):
    """VCAU-Net v2's objective plus the inverse-consistency term.

    Args:
        Everything ``VCAUNetV2Loss`` accepts, plus:
        lambda_inv: weight of ``L_inv``. Defaults to the same order as
            ``lambda_smooth`` because the residual is measured in voxels and is
            therefore an order of magnitude larger than the ``1 - cos`` residual
            that ``lambda_align`` weights. Set to 0 for the
            "bidirectional without inverse consistency" ablation arm, which is
            how the term's contribution is measured rather than assumed.
    """

    def __init__(
        self,
        main_loss: torch.nn.Module,
        aux_loss: torch.nn.Module,
        num_classes: int,
        has_regions: bool,
        ignore_label: Optional[int],
        lambda_aux: float = 0.2,
        lambda_align: float = 0.1,
        lambda_smooth: float = 0.01,
        lambda_inv: float = 0.01,
        lambda_boundary: float = 0.05,
        boundary_warmup_epochs: int = 20,
        aux_scale_decay: float = 0.5,
        distance_radius: int = 6,
    ) -> None:
        super().__init__(
            main_loss=main_loss,
            aux_loss=aux_loss,
            num_classes=num_classes,
            has_regions=has_regions,
            ignore_label=ignore_label,
            lambda_aux=lambda_aux,
            lambda_align=lambda_align,
            lambda_smooth=lambda_smooth,
            lambda_boundary=lambda_boundary,
            boundary_warmup_epochs=boundary_warmup_epochs,
            aux_scale_decay=aux_scale_decay,
            distance_radius=distance_radius,
        )
        if lambda_inv < 0:
            raise ValueError(f"lambda_inv must be non-negative, got {lambda_inv}.")
        self.lambda_inv = float(lambda_inv)

    def forward(
        self,
        output: Union[dict, list, torch.Tensor],
        target,
        return_components: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """Combine every term.

        ``output`` carries an ``"inv"`` entry only for the bidirectional arms; the
        one-way arm produces none and this term is skipped, which is what makes a
        missing term visible as a *zero* contribution rather than as an error.
        """
        total, components = super().forward(output, target, return_components=True)

        if isinstance(output, dict):
            inv_terms = output.get("inv") or []
            if self.lambda_inv > 0 and inv_terms:
                components["inv"] = torch.stack([term.float() for term in inv_terms]).mean()
                total = total + self.lambda_inv * components["inv"]

        if return_components:
            return total, components
        return total
