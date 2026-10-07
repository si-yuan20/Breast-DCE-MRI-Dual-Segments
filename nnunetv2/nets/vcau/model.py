"""VCAU-Net v2: selective residual alignment + uncertainty-disagreement fusion.

Dual-phase DCE-MRI breast tumour segmentation. Input is always a two-channel
volume ``[B, 2, D, H, W]`` with channel 0 = C2 and channel 1 = C5; the output is
``[B, num_classes, D, H, W]`` logits (no sigmoid -- probability handling stays
with nnU-Net's loss and inference code).

What changed from v1
--------------------
======================  ==========================  =============================
                        v1                          v2
======================  ==========================  =============================
State-space bottleneck  VolumetricSSMBottleneck    removed entirely
3D-SPCA                 all six stages              stages 2, 3 only
Fusion                  all six stages, 3C -> C     UGCFv2 at 2,3,4; lightweight
                        dense stack                 1x1x1 residual fusion at 0,1,5
Reliability             relative ratio only         relative + joint + disagreement
Alignment weight        mean of the two R           sqrt(R2 * R5) (detached)
SPCA + fusion params    ~18.4M                      ~4M
======================  ==========================  =============================

The three v1 problems this addresses:

* **SSM was measuring nothing.** At the bottleneck the feature is ``1x2x2`` --
  sequence lengths of 1, 2 and 2. There is no long-range dependency to model
  there, so the state-space block could not have been doing what the paper would
  claim it does.
* **SPCA was applied where it cannot help.** At stage 0/1 the features are
  textural, so contrast uptake can be mistaken for displacement; at stage 5 there
  is no spatial extent for a flow field to act on.
* **UGCF could not tell "both reliable" from "both unreliable"** because
  ``R2/(R2+R5)`` depends only on ``U5 - U2``. See
  :mod:`nnunetv2.nets.vcau_v2.ugcf` for the replacement.

Reuse contract
--------------
The encoder and decoder are nnU-Net's own ``PlainConvEncoder`` and
``UNetDecoder``: stage count, channel widths, kernels, strides, normalisation and
decoder depth all come from the plans. VCAU-Net v2 therefore has exactly the
topology of the nnU-Net baseline and differs only in the cross-phase components,
which is what makes the ablation a controlled comparison rather than an
architecture swap.

The decoder receives the *fused* skips, never the raw C2/C5 features, and its
``deep_supervision`` flag is exposed at ``self.decoder.deep_supervision`` because
``nnUNetTrainer`` toggles it there.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder

from nnunetv2.nets.vcau.losses import cosine_consistency_loss, total_variation_smoothness_loss
from nnunetv2.nets.vcau_v2.blocks import LightweightCrossPhaseFusion3D, PhaseSpecificStem3D
from nnunetv2.nets.vcau_v2.spca import (
    SPCA_MAX_OFFSETS,
    SPCAv2,
    flow_statistics,
    resolve_max_offsets,
)
from nnunetv2.nets.vcau_v2.ugcf import AuxSegHead3D, PhaseUncertainty, UGCFv2

__all__ = ["VCAUNetV2", "SPCA_STAGES", "UGCF_STAGES", "SPCA_MAX_OFFSETS", "INPUT_MODES", "FUSION_MODES"]

INPUT_MODES = ("dual", "c2", "c5")
FUSION_MODES = ("ugcfv2", "lightweight")

#: How 3D-SPCA's alignment weight is derived from the two reliability maps.
#: ``geometric_mean`` = ``sqrt(R2 * R5)`` (the default): gentler than a product, so
#: a voxel where only one phase is confident still contributes to the alignment.
#: ``min`` = ``min(R2, R5)``: stricter -- a voxel counts only as far as the *worse*
#: phase is trustworthy. Both are detached; which is more stable over training is
#: an empirical question, so both are implemented and there is an ablation arm for
#: each rather than a guess baked into the code.
ALIGN_WEIGHT_MODES = ("geometric_mean", "min")

#: Stage policy. Fixed by design (see the module docstring), centralised so it is
#: never restated as ``if stage == 2`` somewhere deep in the forward pass.
SPCA_STAGES: Tuple[int, ...] = (2, 3)
UGCF_STAGES: Tuple[int, ...] = (2, 3, 4)

#: The stage policy assumes the nnU-Net 3d_fullres schedule (stride (1,1,1)
#: followed by 2x downsampling), where stage 5 is the ``1x2x2`` bottleneck for a
#: 64x160x160 patch. Fewer stages would silently shift which scale "stage 2"
#: means, so the network refuses to build instead.
MIN_STAGES = 6


class VCAUNetV2(nn.Module):
    """Dual-phase DCE-MRI segmentation network (plans-driven, 3D only).

    Args:
        input_channels: must be 2 for ``input_mode='dual'`` (channel 0 = C2,
            channel 1 = C5). Single-phase ablation modes still consume the
            two-channel tensor and ignore the other channel, so the data pipeline
            is identical across every ablation arm.
        num_classes: ``LabelManager.num_segmentation_heads`` (2: background + tumour).
        topology arguments (``n_stages`` ... ``n_conv_per_stage_decoder``): nnU-Net
            plans entries, passed through to the shared encoder and the decoder.
            Nothing is hardcoded to a dataset.
        deep_supervision: whether the decoder returns the logits pyramid.
        input_mode: ``dual`` / ``c2`` / ``c5``.
        fusion_mode: ``ugcfv2`` (uncertainty-disagreement guided, at
            ``ugcf_stages``) or ``lightweight`` (plain ``1x1x1`` residual fusion
            everywhere -- the ``-UGCF`` ablation).
        use_spca: build and apply 3D-SPCA. Disabling it removes the alignment path
            *and* ``L_align``/``L_smooth``, and fuses the unaligned C5 features.
        use_joint_reliability / use_disagreement_gate: ablation switches for the two
            factors of ``G = g_u * g_d``. Setting one to False replaces that factor
            with 1 and changes nothing else, so each ablation moves one variable.
        spca_stages / ugcf_stages / spca_max_offsets: stage policy and per-stage
            displacement bounds; defaults are the module-level constants.
        tau_phase / tau_joint / gate_hidden: UGCFv2 hyper-parameters.
        fusion_hidden_ratio: capacity knob for the lightweight fusion operator. The
            default (1.0) gives the single-``1x1x1`` operator; the capacity-matched
            control arm raises it to match the full model's parameter count.
        padding_mode: ``grid_sample`` padding for the warp.
    """

    #: Explicit contract for the test suite and the paper: v2 does not use a
    #: state-space model at any stage. ``VolumetricSSMBottleneck`` is neither
    #: imported nor instantiated here.
    USES_SSM = False

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
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------ validation
        if align_weight_mode not in ALIGN_WEIGHT_MODES:
            raise ValueError(
                f"align_weight_mode must be one of {ALIGN_WEIGHT_MODES}, got "
                f"{align_weight_mode!r}."
            )
        if input_mode not in INPUT_MODES:
            raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}.")
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}, got {fusion_mode!r}.")
        if conv_op is not nn.Conv3d:
            raise ValueError(
                f"VCAU-Net v2 is a 3D network; the plans configuration must use a 3D "
                f"convolution, got {getattr(conv_op, '__name__', conv_op)}."
            )
        if input_mode == "dual" and input_channels != 2:
            raise ValueError(
                f"VCAU-Net v2 requires exactly two input channels: C2 and C5. Got {input_channels}."
            )
        if input_mode == "c5" and input_channels < 2:
            raise ValueError(
                f"input_mode='c5' reads channel 1 (C5) but the data has only {input_channels} "
                f"channel(s)."
            )
        if input_channels < 1:
            raise ValueError(f"input_channels must be positive, got {input_channels}.")
        if fusion_hidden_ratio <= 0:
            raise ValueError(f"fusion_hidden_ratio must be positive, got {fusion_hidden_ratio}.")

        n_stages = int(n_stages)
        if n_stages < MIN_STAGES:
            raise ValueError(
                f"VCAU-Net v2's stage policy (SPCA at {tuple(SPCA_STAGES)}, UGCFv2 at "
                f"{tuple(UGCF_STAGES)}, lightweight fusion elsewhere incl. the bottleneck at "
                f"stage {MIN_STAGES - 1}) is defined for at least {MIN_STAGES} stages; the plans "
                f"provide {n_stages}. Fewer stages would silently shift which scale 'stage 2' "
                f"refers to."
            )
        self.spca_stages = tuple(int(s) for s in spca_stages)
        self.ugcf_stages = tuple(int(s) for s in ugcf_stages)
        for label, stages in (("spca_stages", self.spca_stages), ("ugcf_stages", self.ugcf_stages)):
            for stage in stages:
                if not 0 <= stage < n_stages:
                    raise ValueError(
                        f"{label} contains stage {stage}, outside [0, {n_stages - 1}]."
                    )
        if not set(self.spca_stages).issubset(self.ugcf_stages):
            raise ValueError(
                f"Every SPCA stage needs a reliability map to weight its alignment loss, so "
                f"spca_stages {self.spca_stages} must be a subset of ugcf_stages "
                f"{self.ugcf_stages}."
            )

        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        features_per_stage = [int(c) for c in features_per_stage]
        n_conv_per_stage = list(n_conv_per_stage)
        n_conv_per_stage_decoder = list(n_conv_per_stage_decoder)
        strides = [tuple(int(v) for v in s) for s in strides]

        if len(features_per_stage) != n_stages:
            raise ValueError(
                f"features_per_stage has {len(features_per_stage)} entries but n_stages={n_stages}."
            )
        if len(strides) != n_stages:
            raise ValueError(
                f"strides has {len(strides)} entries but n_stages={n_stages}. The decoder derives "
                f"its number of deep-supervision outputs from these strides, so they must agree "
                f"with the plans' pool_op_kernel_sizes."
            )
        if len(n_conv_per_stage_decoder) != n_stages - 1:
            raise ValueError(
                f"n_conv_per_stage_decoder must have n_stages-1={n_stages - 1} entries, got "
                f"{len(n_conv_per_stage_decoder)}."
            )

        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
        self.input_mode = input_mode
        self.fusion_mode = fusion_mode
        self.use_spca = bool(use_spca)
        self.tau_phase = float(tau_phase)
        self.deep_supervision = bool(deep_supervision)
        self.fusion_hidden_ratio = float(fusion_hidden_ratio)
        self.align_weight_mode = align_weight_mode

        dual = input_mode == "dual"
        # SPCA needs both phases; reliability is needed whenever UGCFv2 runs or
        # whenever SPCA's alignment weight has to come from somewhere.
        self.spca_active = dual and self.use_spca
        self.ugcf_active = dual and fusion_mode == "ugcfv2"
        self.needs_reliability = dual and (self.ugcf_active or self.spca_active)
        self.max_offsets = resolve_max_offsets(
            self.spca_stages, SPCA_MAX_OFFSETS if spca_max_offsets is None else spca_max_offsets
        )

        # ---------------------------------------------------------------- stems
        stem_kwargs = dict(
            conv_op=conv_op,
            out_channels=features_per_stage[0],
            kernel_size=kernel_sizes[0],
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
        )
        self.stem_c2 = (
            PhaseSpecificStem3D(in_channels=1, **stem_kwargs) if input_mode in ("dual", "c2") else None
        )
        self.stem_c5 = (
            PhaseSpecificStem3D(in_channels=1, **stem_kwargs) if input_mode in ("dual", "c5") else None
        )

        # -------------------------------------------------- shared 3D encoder
        # ONE instance, called once per phase: the deep parameters are shared by
        # construction, there is no encoder_c2 / encoder_c5 pair.
        self.encoder = PlainConvEncoder(
            input_channels=features_per_stage[0],
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            nonlin_first=False,
            pool=pool,
        )

        # ------------------------------------------------- per-scale components
        self.uncertainty = PhaseUncertainty()
        self.spca = nn.ModuleList()
        self.ugcf = nn.ModuleList()
        self.light_fusion = nn.ModuleList()
        self.aux_heads_c2 = nn.ModuleList()
        self.aux_heads_c5 = nn.ModuleList()

        for stage, channels in enumerate(features_per_stage):
            runs_spca = self.spca_active and stage in self.spca_stages
            self.spca.append(
                SPCAv2(
                    conv_op,
                    channels,
                    self.max_offsets[stage],
                    kernel_size=kernel_sizes[stage],
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    padding_mode=padding_mode,
                )
                if runs_spca
                else None
            )

            runs_ugcf = self.ugcf_active and stage in self.ugcf_stages
            self.ugcf.append(
                UGCFv2(
                    conv_op,
                    channels,
                    kernel_size=kernel_sizes[stage],
                    tau_phase=tau_phase,
                    tau_joint=tau_joint,
                    gate_hidden=gate_hidden,
                    use_joint_reliability=use_joint_reliability,
                    use_disagreement_gate=use_disagreement_gate,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                )
                if runs_ugcf
                else None
            )

            # Every stage the decoder reads needs a fused skip. Stages outside the
            # UGCFv2 policy get the light operator; so does every stage when
            # fusion_mode='lightweight'.
            self.light_fusion.append(
                LightweightCrossPhaseFusion3D(
                    conv_op,
                    channels,
                    norm_op,
                    norm_op_kwargs,
                    hidden_ratio=fusion_hidden_ratio,
                )
                if (dual and not runs_ugcf)
                else None
            )

            has_aux = self.needs_reliability and stage in self.ugcf_stages
            self.aux_heads_c2.append(AuxSegHead3D(conv_op, channels, num_classes) if has_aux else None)
            self.aux_heads_c5.append(AuxSegHead3D(conv_op, channels, num_classes) if has_aux else None)

        # ------------------------------------------------------------- decoder
        self.decoder = UNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            nonlin_first=False,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            conv_bias=conv_bias,
        )

    # ------------------------------------------------------------------ utils
    def set_deep_supervision(self, enabled: bool) -> None:
        """Toggle the decoder's deep-supervision pyramid.

        Mirrors ``nnUNetTrainer.set_deep_supervision_enabled`` (which writes
        ``self.decoder.deep_supervision``); provided so the network can also be
        driven directly, e.g. from tests.
        """
        self.deep_supervision = bool(enabled)
        self.decoder.deep_supervision = bool(enabled)

    def _encode(self, x: torch.Tensor) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        """Stem + shared encoder, applied to each phase with the same weights."""
        skips_c2 = self.encoder(self.stem_c2(x[:, :1])) if self.stem_c2 is not None else None
        skips_c5 = self.encoder(self.stem_c5(x[:, 1:2])) if self.stem_c5 is not None else None
        return skips_c2, skips_c5

    def reliability_maps(
        self, logits_c2: torch.Tensor, logits_c5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``q2``/``q5`` for a pair of auxiliary logits, both detached.

        Used by the ``-UGCF`` arm, where SPCA is still active and its alignment
        weight still has to come from somewhere even though no UGCFv2 module was
        built.
        """
        _, uncertainty_c2 = self.uncertainty(logits_c2)
        _, uncertainty_c5 = self.uncertainty(logits_c5)
        return (
            torch.exp(-uncertainty_c2 / self.tau_phase),
            torch.exp(-uncertainty_c5 / self.tau_phase),
        )

    # ---------------------------------------------------------------- features
    def _build_features(
        self, x: torch.Tensor, keep_details: bool
    ) -> Tuple[List[torch.Tensor], Dict[str, list], Dict[str, torch.Tensor]]:
        """Fused skip pyramid plus per-scale supervision signals.

        Returns:
            fused skips (bottleneck last, ready for the decoder),
            per-scale auxiliary logits / alignment / smoothness terms,
            scalar diagnostics.
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
        diagnostics: Dict[str, torch.Tensor] = {}

        for stage in range(self.n_stages):
            if single_phase is not None:
                # Single-phase ablation: the shared encoder is unchanged, so this
                # is "VCAU-Net v2 on one phase", not a different model.
                fused.append(single_phase[stage])
                continue

            f2 = skips_c2[stage]
            f5 = skips_c5[stage]

            spca = self.spca[stage]
            if spca is not None:
                aligned = spca(f2, f5)
                f5_aligned, z2, z5_aligned, flow = aligned
                smooth_terms.append(total_variation_smoothness_loss(flow))
                if keep_details:
                    for name, value in flow_statistics(flow).items():
                        diagnostics[f"spca_s{stage}_flow_{name}"] = value
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
                # Both modes are detached; the alignment loss asserts this itself.
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
        }
        return fused, details, diagnostics

    @staticmethod
    def _aggregate_diagnostics(diagnostics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add cross-stage means so the trainer can watch one number per quantity."""
        aggregates: Dict[str, torch.Tensor] = {}
        for name in ("alpha2", "alpha5", "U2", "U5", "D_p", "G", "alpha2_deviation"):
            keys = [
                key
                for key in diagnostics
                if key.startswith("ugcf_s") and key.endswith(f"_{name}_mean")
            ]
            if keys:
                aggregates[f"mean_{name}"] = torch.stack(
                    [diagnostics[key].detach().float() for key in keys]
                ).mean()
        flow_keys = [key for key in diagnostics if key.startswith("spca_s") and key.endswith("_flow_mean")]
        if flow_keys:
            aggregates["mean_flow_magnitude"] = torch.stack(
                [diagnostics[key].detach().float() for key in flow_keys]
            ).mean()
        return aggregates

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
            "stats": diagnostics,
        }

    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, object]]:
        """Forward pass.

        ``return_aux=False`` (the default, and the only mode nnU-Net inference
        uses) returns exactly what the stock nnU-Net architectures return: a list
        of logits when deep supervision is on, a single tensor otherwise.
        ``nnUNetPredictor`` performs ``prediction += torch.flip(network(x), ...)``
        for mirroring, so a non-tensor return here would break inference.
        """
        if x.ndim != 5:
            raise ValueError(f"VCAUNetV2 expects a 5D [B, C, D, H, W] input, got {tuple(x.shape)}.")
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channel(s), got {x.shape[1]}. The dataset "
                f"must provide C2 in channel 0 and C5 in channel 1."
            )
        if return_aux:
            return self.forward_train(x)

        fused, _, _ = self._build_features(x, keep_details=False)
        return self.decoder(fused)

    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> np.int64:
        """Conservative feature-map estimate used by nnU-Net planning utilities."""
        encoder_maps = np.int64(self.encoder.compute_conv_feature_map_size(input_size))
        if self.input_mode == "dual":
            encoder_maps = encoder_maps * 2
        decoder_maps = np.int64(0)
        dims = list(input_size)
        skip_sizes = []
        for stride in self.encoder.strides[:-1]:
            dims = [size // step for size, step in zip(dims, stride)]
            skip_sizes.append(dims)
        for stage, spatial_size in enumerate(skip_sizes[::-1]):
            decoder_maps += self.decoder.stages[stage].compute_conv_feature_map_size(spatial_size)
            channels = self.encoder.output_channels[-(stage + 2)]
            decoder_maps += np.prod([channels, *spatial_size], dtype=np.int64)
            decoder_maps += np.prod([self.num_classes, *spatial_size], dtype=np.int64)
        return encoder_maps + decoder_maps
