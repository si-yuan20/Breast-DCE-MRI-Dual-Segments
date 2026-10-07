"""VCAU-Net v3: symmetric cross-phase correspondence, asymmetric C2 reference.

Identical to :class:`~nnunetv2.nets.vcau_v2.model.VCAUNetV2` in every respect
except the alignment operator:

===========================  ==========================  ==========================
                             v2                          v3
===========================  ==========================  ==========================
structural projection psi    1x1x1 ConvNormAct          1x1x1 + depthwise k^3, residual
flow input                   cat(Z2, Z5), 2C            cat(a, b, |a-b|, a*b), 4C
flow directions              C5 -> C2                    C5 -> C2 and C2 -> C5
flow predictor               one instance                one instance, called twice
regularisers                 L_smooth (TV, one field)    L_smooth (TV, both) + L_inv
applies to features          O_5->2                      O_5->2 only
reference frame              C2                          C2 (unchanged)
===========================  ==========================  ==========================

Stage policy (2, 3), the decoder, the shared encoder, UGCFv2 and the exported
logit space are untouched. C2 remains the fixed reference: ``O_2->5`` is a
training-time regularisation signal and never reaches a feature.

Why this is a subclass rather than an edit
------------------------------------------
``model.py`` is frozen so the v2 arms stay reproducible for the paper's ablation
table. That costs one duplicated method (:meth:`VCAUNetV3._build_features`, which
has to handle the 6-field :class:`SPCAv3Output` and append the inverse-consistency
term). Because duplicated control flow is how two arms silently drift apart,
``test/test_vcau_v3.py`` asserts that ``VCAUNetV3`` and ``VCAUNetV2`` produce
**bit-identical** output with SPCA disabled.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from nnunetv2.nets.vcau.losses import cosine_consistency_loss, total_variation_smoothness_loss
from nnunetv2.nets.vcau_v2.model import SPCA_STAGES, UGCF_STAGES, VCAUNetV2
from nnunetv2.nets.vcau_v2.spca_v3 import (
    DESCRIPTOR_MODES,
    SPCAv3,
    flow_statistics,
    inverse_consistency_loss,
)

__all__ = ["VCAUNetV3", "SPCA_V3_STAT_KEYS"]

#: Direction tags used in the diagnostic keys, in the order they are reported.
DIRECTIONS = ("5_to_2", "2_to_5")

#: The six scalars :func:`~nnunetv2.nets.vcau_v2.spca_v3.flow_statistics` returns.
SPCA_V3_STAT_KEYS = (
    "component_abs_mean",
    "component_abs_p95",
    "component_abs_max",
    "mean_magnitude",
    "p95_magnitude",
    "max_magnitude",
)


class VCAUNetV3(VCAUNetV2):
    """Dual-phase DCE-MRI segmentation network with symmetric 3D-SPCA.

    Args:
        Every argument ``VCAUNetV2`` accepts, plus:
        descriptor_mode: ``symmetric`` (4C, the method) or ``concat`` (2C, v2's
            input). The ``concat`` mode is the descriptor-ablation control: it
            keeps the new adapter, the bidirectional predictor and the
            inverse-consistency term while reverting only the descriptor, which
            is what separates "the descriptor helped" from "more parameters
            helped".
        bidirectional: predict ``O_2->5`` as well as ``O_5->2``. ``False`` gives
            the v2 topology with v3's adapter and descriptor.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: Union[int, Sequence[int]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, Sequence[Sequence[int]]],
        strides: Union[int, Sequence[Sequence[int]]],
        n_conv_per_stage: Union[int, Sequence[int]],
        n_conv_per_stage_decoder: Union[int, Sequence[int]],
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[_DropoutNd]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = True,
        pool: str = "conv",
        input_mode: str = "dual",
        fusion_mode: str = "ugcfv2",
        use_spca: bool = True,
        use_joint_reliability: bool = True,
        use_disagreement_gate: bool = True,
        spca_stages: Sequence[int] = SPCA_STAGES,
        ugcf_stages: Sequence[int] = UGCF_STAGES,
        spca_max_offsets: Optional[Dict[int, int]] = None,
        tau_phase: float = 0.5,
        tau_joint: float = 0.25,
        gate_hidden: int = 8,
        fusion_hidden_ratio: float = 1.0,
        padding_mode: str = "border",
        align_weight_mode: str = "geometric_mean",
        descriptor_mode: str = "symmetric",
        bidirectional: bool = True,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            pool=pool,
            input_mode=input_mode,
            fusion_mode=fusion_mode,
            use_spca=use_spca,
            use_joint_reliability=use_joint_reliability,
            use_disagreement_gate=use_disagreement_gate,
            spca_stages=spca_stages,
            ugcf_stages=ugcf_stages,
            spca_max_offsets=spca_max_offsets,
            tau_phase=tau_phase,
            tau_joint=tau_joint,
            gate_hidden=gate_hidden,
            fusion_hidden_ratio=fusion_hidden_ratio,
            padding_mode=padding_mode,
            align_weight_mode=align_weight_mode,
        )

        if descriptor_mode not in DESCRIPTOR_MODES:
            raise ValueError(
                f"descriptor_mode must be one of {DESCRIPTOR_MODES}, got {descriptor_mode!r}."
            )
        self.descriptor_mode = descriptor_mode
        self.bidirectional = bool(bidirectional)

        # The parent built SPCAv2 modules; replace exactly those with SPCAv3,
        # keeping the same stage policy and per-stage displacement bounds.
        for stage, module in enumerate(self.spca):
            if module is None:
                continue
            self.spca[stage] = SPCAv3(
                conv_op,
                self.features_per_stage[stage],
                self.max_offsets[stage],
                kernel_size=kernel_sizes[stage],
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                padding_mode=padding_mode,
                descriptor_mode=descriptor_mode,
                bidirectional=bidirectional,
            )

        if self.spca_active:
            stages = sorted(int(s) for s in self.spca_stages)
            self.spca_kind = "symmetric" if descriptor_mode == "symmetric" else "concat"
            layout = (
                f"{self.spca_kind} descriptor "
                f"({4 if descriptor_mode == 'symmetric' else 2}C) -> "
                f"{'bidirectional' if bidirectional else 'one-way'} flow at stages {stages}"
            )
        else:
            self.spca_kind = "off"
            layout = "OFF"
        self.spca_layout = layout

    # ------------------------------------------------------------------ utils
    def spca_flow_predictors(self) -> List[nn.Module]:
        """The single per-stage predictor instances, for gradient/test inspection.

        There is exactly one ``flow_predictor`` per SPCA stage, shared by both
        directions -- so this list has one entry per active stage, not two.
        """
        return [module.flow_predictor for module in self.spca if module is not None]

    @staticmethod
    def _flows_of(aligned) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``(O_5->2, O_2->5)`` from an :class:`SPCAv3Output`."""
        return aligned.flow_5_to_2, aligned.flow_2_to_5

    @staticmethod
    def _record_flow_diagnostics(
        diagnostics: Dict[str, torch.Tensor],
        stage: int,
        direction: str,
        flow: torch.Tensor,
    ) -> None:
        """Per-stage, per-direction magnitude diagnostics.

        Kept separate per direction on purpose: a reverse field that never moves
        while the forward one does is a specific, diagnosable failure of the
        inverse-consistency term, and a single merged average would hide it.
        """
        for name, value in flow_statistics(flow).items():
            diagnostics[f"spca_s{stage}_{direction}_{name}"] = value

    @staticmethod
    def _aggregate_diagnostics(diagnostics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Cross-stage means, per direction and merged.

        ``mean_flow_magnitude`` keeps the key the trainer's collapse detector
        already reads; the two per-direction keys are what make a one-sided
        collapse visible.
        """
        aggregates = VCAUNetV2._aggregate_diagnostics(diagnostics)

        def _mean_of(suffix: str) -> Optional[torch.Tensor]:
            keys = [
                key
                for key in diagnostics
                if key.startswith("spca_s") and key.endswith(suffix)
            ]
            if not keys:
                return None
            return torch.stack([diagnostics[key].detach().float() for key in keys]).mean()

        for direction in DIRECTIONS:
            value = _mean_of(f"_{direction}_mean_magnitude")
            if value is not None:
                aggregates[f"mean_flow_{direction}_magnitude"] = value
        merged = _mean_of("_mean_magnitude")
        if merged is not None:
            aggregates["mean_flow_magnitude"] = merged
        return aggregates

    # ---------------------------------------------------------------- features
    def _build_features(
        self, x: torch.Tensor, keep_details: bool
    ) -> Tuple[List[torch.Tensor], Dict[str, list], Dict[str, torch.Tensor]]:
        """Fused skip pyramid plus per-scale supervision signals.

        Mirrors ``VCAUNetV2._build_features`` with two differences: the SPCA
        output carries six fields instead of four, and the reverse flow
        contributes a second smoothness term plus the inverse-consistency term.

        Per stage the smoothness contribution is ``TV(O_5->2) + TV(O_2->5)`` as a
        *single* entry, so ``lambda_smooth`` keeps the same meaning it has in v2.
        """
        skips_c2, skips_c5 = self._encode(x)
        if self.stem_c5 is None:
            single_phase: Optional[List[torch.Tensor]] = skips_c2
        elif self.stem_c2 is None:
            single_phase = skips_c5
        else:
            single_phase = None

        fused: List[torch.Tensor] = []
        aux_c2: List[torch.Tensor] = []
        aux_c5: List[torch.Tensor] = []
        align_terms: List[torch.Tensor] = []
        smooth_terms: List[torch.Tensor] = []
        inv_terms: List[torch.Tensor] = []
        diagnostics: Dict[str, torch.Tensor] = {}

        for stage in range(self.n_stages):
            if single_phase is not None:
                fused.append(single_phase[stage])
                continue

            f2 = skips_c2[stage]
            f5 = skips_c5[stage]

            spca = self.spca[stage]
            if spca is not None:
                aligned = spca(f2, f5)
                f5_aligned = aligned.f5_aligned
                z2 = aligned.z2
                z5_aligned = aligned.z5_aligned
                flow_5_to_2, flow_2_to_5 = self._flows_of(aligned)

                smoothness = total_variation_smoothness_loss(flow_5_to_2)
                if flow_2_to_5 is not None:
                    smoothness = smoothness + total_variation_smoothness_loss(flow_2_to_5)
                    inv_terms.append(
                        inverse_consistency_loss(flow_5_to_2, flow_2_to_5, spca.warp)
                    )
                smooth_terms.append(smoothness)

                if keep_details:
                    self._record_flow_diagnostics(diagnostics, stage, "5_to_2", flow_5_to_2)
                    if flow_2_to_5 is not None:
                        self._record_flow_diagnostics(diagnostics, stage, "2_to_5", flow_2_to_5)
            else:
                f5_aligned = f5
                z2 = z5_aligned = None

            ugcf = self.ugcf[stage]
            head_c2 = self.aux_heads_c2[stage]
            reliability_c2 = reliability_c5 = None

            if head_c2 is not None:
                logits_c2 = head_c2(f2)
                logits_c5 = self.aux_heads_c5[stage](f5_aligned)
                aux_c2.append(logits_c2)
                aux_c5.append(logits_c5)
                if ugcf is not None:
                    ugcf_output = ugcf(f2, f5_aligned, logits_c2, logits_c5)
                    fused.append(ugcf_output.fused)
                    reliability_c2 = ugcf_output.reliability_c2
                    reliability_c5 = ugcf_output.reliability_c5
                    if keep_details:
                        for name, value in ugcf_output.statistics.items():
                            diagnostics[f"ugcf_s{stage}_{name}"] = value
                else:
                    reliability_c2, reliability_c5 = self.reliability_maps(logits_c2, logits_c5)
                    fused.append(self.light_fusion[stage](f2, f5_aligned))
            else:
                fused.append(self.light_fusion[stage](f2, f5_aligned))

            if spca is not None and reliability_c2 is not None:
                # Both modes are detached; cosine_consistency_loss asserts it.
                if self.align_weight_mode == "geometric_mean":
                    weight = torch.sqrt(reliability_c2 * reliability_c5)
                else:
                    weight = torch.minimum(reliability_c2, reliability_c5)
                align_terms.append(cosine_consistency_loss(z2, z5_aligned, weight))

        if keep_details and diagnostics:
            diagnostics.update(self._aggregate_diagnostics(diagnostics))

        details = {
            "aux_c2": aux_c2,
            "aux_c5": aux_c5,
            "align": align_terms,
            "smooth": smooth_terms,
            "inv": inv_terms,
        }
        return fused, details, diagnostics

    # ---------------------------------------------------------------- forward
    def forward_train(self, x: torch.Tensor) -> Dict[str, object]:
        """Training forward: segmentation logits plus every supervision signal."""
        fused, details, diagnostics = self._build_features(x, keep_details=True)
        return {
            "seg": self.decoder(fused),
            "aux_c2": details["aux_c2"],
            "aux_c5": details["aux_c5"],
            "align": details["align"],
            "smooth": details["smooth"],
            "inv": details["inv"],
            "stats": diagnostics,
        }
