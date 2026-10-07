"""nnU-Net v2 trainer for VCAU-Net (dual-phase DCE-MRI breast tumour segmentation).

Run with::

    nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerVCAUNet

Everything that is not specific to VCAU-Net is inherited unchanged: planning,
preprocessing, the augmentation pipeline, the optimiser and LR schedule, AMP,
distributed training, deep supervision, checkpointing, sliding-window inference
and prediction export.

Override list and why each one is unavoidable
---------------------------------------------
``__init__``                    prints the configuration summary (required by the
                                reporting spec) and fails fast on channel errors.
``build_network_architecture``  classmethod so ablation subclasses can select
                                components through class attributes while the
                                inference path (which calls this on the class,
                                not on an instance) keeps working.
``_build_loss``                 wraps the stock nnU-Net loss in the compound
                                objective; the main term itself is reused as-is.
``train_step``                  needs the auxiliary output of the network.
``validation_step``             same, and must keep the stock return contract.
``on_train_epoch_end``          reports the per-term losses and the alignment /
                                reliability diagnostics. nnU-Net's LocalLogger
                                asserts on a fixed key whitelist, so custom
                                metrics cannot go through ``self.logger.log``.
``get_training_transforms``     synchronises intensity augmentation across the
                                C2/C5 channels (see the docstring below).
``_do_i_compile``               eager execution; the training forward returns
                                structured output and is not compile-friendly.

Deliberately *not* overridden: ``initialize``, ``configure_optimizers``,
``save_checkpoint``, ``load_checkpoint``, DDP wiring, AMP, and
``set_deep_supervision_enabled`` -- the last one works unchanged because the
network exposes ``self.decoder.deep_supervision``, which is exactly what the
base implementation toggles.
"""

from __future__ import annotations

import pydoc
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn

from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from nnunetv2.nets.vcau.losses import VCAUCompoundLoss
from nnunetv2.nets.vcau.model import VCAUNet3D
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss, get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerVCAUNet(nnUNetTrainer):
    """VCAU-Net on the nnU-Net v2 base chain.

    The class attributes below are the single place where VCAU-Net's own
    configuration lives. Ablation trainers subclass this class and override
    nothing but these flags, so no training logic is ever duplicated.
    """

    # Network composition (overridden by the ablation trainers).
    INPUT_MODE = "dual"
    FUSION_MODE = "ugcf"
    USE_SPCA = True
    USE_UGCF = True
    USE_SSM = True

    # 3D-SPCA.
    SPCA_MIN_STAGE = 0
    MAX_OFFSETS = None

    # Reliability.
    TAU = 0.5

    # State-space bottleneck.
    SSM_D_STATE = 8

    # Loss weights.
    LAMBDA_AUX = 0.2
    LAMBDA_BOUNDARY = 0.1
    LAMBDA_ALIGN = 0.1
    LAMBDA_SMOOTH = 0.01
    AUX_SCALE_DECAY = 0.5
    BOUNDARY_DISTANCE_RADIUS = 6

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self._print_configuration_summary()

    # -------------------------------------------------------------- reporting
    def _print_configuration_summary(self) -> None:
        configuration = self.configuration_manager
        # The convolution kernels live inside the plans' architecture kwargs, not
        # as a top-level ConfigurationManager property.
        architecture_kwargs = configuration.network_arch_init_kwargs
        conv_kernels = architecture_kwargs.get("kernel_sizes", "?")
        features = architecture_kwargs.get("features_per_stage", "?")
        lines = [
            "",
            "=" * 68,
            "[VCAU-Net] Volumetric Cross-phase Alignment and Uncertainty-aware Network",
            "=" * 68,
            "  Input channels:     2",
            "  C2 channel:         0",
            "  C5 channel:         1",
            f"  input_mode:         {self.INPUT_MODE}",
            f"  Patch size:         {tuple(configuration.patch_size)}",
            f"  Batch size:         {configuration.batch_size}",
            f"  Stages:             {len(configuration.pool_op_kernel_sizes)}",
            f"  Pool strides:       {configuration.pool_op_kernel_sizes}",
            f"  Conv kernels:       {conv_kernels}",
            f"  Channels:           {features}",
            f"  3D-SPCA:            {'ON' if self.USE_SPCA else 'OFF'}"
            f" (from stage {self.SPCA_MIN_STAGE})",
            f"  UGCF:               {'ON' if self.USE_UGCF else 'OFF'}"
            f" (fusion_mode={self.FUSION_MODE})",
            f"  SSM bottleneck:     {'ON' if self.USE_SSM else 'OFF'}"
            f" (d_state={self.SSM_D_STATE})",
            f"  tau:                {self.TAU}",
            f"  max_offsets:        {self.MAX_OFFSETS if self.MAX_OFFSETS else 'auto (4,3,2,1,...)'}",
            f"  Deep supervision:   {self.enable_deep_supervision}",
            f"  lambda_aux:         {self.LAMBDA_AUX}",
            f"  lambda_boundary:    {self.LAMBDA_BOUNDARY}",
            f"  lambda_align:       {self.LAMBDA_ALIGN}",
            f"  lambda_smooth:      {self.LAMBDA_SMOOTH}",
            f"  Optimizer:          SGD(lr={self.initial_lr}, wd={self.weight_decay}, momentum=0.99, nesterov)",
            f"  Epochs:             {self.num_epochs}",
            "  Output:             logits [B, num_classes, D, H, W]; softmax is applied by the loss",
            "=" * 68,
        ]
        for line in lines:
            self.print_to_log_file(line)

    # ------------------------------------------------------------ architecture
    @classmethod
    def build_network_architecture(
        cls,
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """Build VCAU-Net from the nnU-Net plans.

        A classmethod rather than a staticmethod so ablation subclasses can pick
        their components from class attributes, while ``nnUNetPredictor`` -- which
        calls this on the class object, not on an instance -- still works.

        All topology (stage count, channel widths, kernels, strides, decoder
        depth) comes from ``arch_init_kwargs``, i.e. from the plans. Nothing is
        hardcoded, and no plans file is written to.
        """
        if num_input_channels != 2:
            raise ValueError(
                f"VCAU-Net requires exactly two input channels: C2 and C5. The plans/dataset "
                f"provide {num_input_channels}. Channel 0 must be C2 and channel 1 must be C5."
            )

        architecture_kwargs = dict(arch_init_kwargs)
        for key in arch_init_kwargs_req_import:
            if architecture_kwargs.get(key) is not None:
                resolved = pydoc.locate(architecture_kwargs[key])
                if resolved is None:
                    raise ImportError(
                        f"Could not resolve the plans entry {key}={architecture_kwargs[key]!r}."
                    )
                architecture_kwargs[key] = resolved

        n_conv_per_stage = architecture_kwargs.get("n_conv_per_stage")
        if n_conv_per_stage is None:
            n_conv_per_stage = architecture_kwargs.get("n_blocks_per_stage")
        required = ("n_stages", "features_per_stage", "conv_op", "kernel_sizes", "strides")
        missing = [key for key in required if key not in architecture_kwargs]
        if missing or n_conv_per_stage is None:
            raise ValueError(
                f"Incomplete nnU-Net architecture plans; missing {missing} "
                f"(n_conv_per_stage/n_blocks_per_stage={n_conv_per_stage})."
            )
        if architecture_kwargs["conv_op"] is not nn.Conv3d:
            raise ValueError(
                "VCAU-Net only supports the 3D configuration; got a conv_op of "
                f"{architecture_kwargs['conv_op']}. Run 3d_fullres."
            )

        network = VCAUNet3D(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            n_stages=architecture_kwargs["n_stages"],
            features_per_stage=architecture_kwargs["features_per_stage"],
            conv_op=architecture_kwargs["conv_op"],
            kernel_sizes=architecture_kwargs["kernel_sizes"],
            strides=architecture_kwargs["strides"],
            n_conv_per_stage=n_conv_per_stage,
            n_conv_per_stage_decoder=architecture_kwargs["n_conv_per_stage_decoder"],
            conv_bias=architecture_kwargs.get("conv_bias", True),
            norm_op=architecture_kwargs.get("norm_op"),
            norm_op_kwargs=architecture_kwargs.get("norm_op_kwargs"),
            dropout_op=architecture_kwargs.get("dropout_op"),
            dropout_op_kwargs=architecture_kwargs.get("dropout_op_kwargs"),
            nonlin=architecture_kwargs.get("nonlin"),
            nonlin_kwargs=architecture_kwargs.get("nonlin_kwargs"),
            deep_supervision=enable_deep_supervision,
            pool=architecture_kwargs.get("pool", "conv"),
            input_mode=cls.INPUT_MODE,
            fusion_mode=cls.FUSION_MODE,
            use_spca=cls.USE_SPCA,
            use_ugcf=cls.USE_UGCF,
            use_ssm=cls.USE_SSM,
            spca_min_stage=cls.SPCA_MIN_STAGE,
            max_offsets=cls.MAX_OFFSETS,
            tau=cls.TAU,
            ssm_d_state=cls.SSM_D_STATE,
        )
        network.apply(InitWeights_He(1e-2))
        return network

    # ------------------------------------------------------------------- loss
    def _build_loss(self):
        """Stock nnU-Net segmentation loss plus VCAU-Net's own terms.

        The main term is literally ``super()._build_loss()``: the same
        ``DC_and_CE_loss`` with the same ``MemoryEfficientSoftDiceLoss`` and the
        same deep-supervision wrapper the nnU-Net baseline optimises, so any
        performance difference cannot come from a re-implemented Dice.
        """
        main_loss = super()._build_loss()
        if self.label_manager.has_regions:
            raise ValueError(
                "VCAU-Net implements the two-class tumour formulation; region-based training "
                "(label_manager.has_regions) is not supported by its boundary/auxiliary terms."
            )
        aux_loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        loss = VCAUCompoundLoss(
            main_loss=main_loss,
            aux_loss=aux_loss,
            num_classes=self.label_manager.num_segmentation_heads,
            has_regions=self.label_manager.has_regions,
            ignore_label=self.label_manager.ignore_label,
            lambda_aux=self.LAMBDA_AUX,
            lambda_boundary=self.LAMBDA_BOUNDARY,
            lambda_align=self.LAMBDA_ALIGN,
            lambda_smooth=self.LAMBDA_SMOOTH,
            aux_scale_decay=self.AUX_SCALE_DECAY,
            distance_radius=self.BOUNDARY_DISTANCE_RADIUS,
        )
        self.print_to_log_file(
            f"[VCAU-Net] loss weights: aux={self.LAMBDA_AUX} boundary={self.LAMBDA_BOUNDARY} "
            f"align={self.LAMBDA_ALIGN} smooth={self.LAMBDA_SMOOTH}"
        )
        return loss

    # --------------------------------------------------------------- training
    @staticmethod
    def get_training_transforms(*args, **kwargs):
        """Keep intensity augmentation synchronised across C2 and C5.

        The stock nnU-Net 3D pipeline samples brightness, contrast, gamma, noise,
        blur and low-resolution simulation *independently per channel*. That is
        the right behaviour for unrelated modalities such as CT and PET, but for
        two phases of the same DCE acquisition it actively destroys the signal
        VCAU-Net is built to model: independent per-channel intensity jitter
        invents or erases enhancement differences, and 3D-SPCA/UGCF would then be
        trained to align and fuse features that no longer correspond to the same
        kinetics.

        Spatial transforms are already applied jointly to all channels by the
        nnU-Net pipeline, so only the intensity family needs fixing here.
        """
        transform = nnUNetTrainer.get_training_transforms(*args, **kwargs)

        def synchronize(node) -> None:
            if hasattr(node, "synchronize_channels"):
                node.synchronize_channels = True
            for child in getattr(node, "transforms", []) or []:
                synchronize(child)
            child = getattr(node, "transform", None)
            if child is not None:
                synchronize(child)

        synchronize(transform)
        return transform

    def train_step(self, batch: dict) -> dict:
        """One optimisation step.

        Mirrors ``nnUNetTrainer.train_step`` exactly -- same autocast handling,
        same gradient clipping at 12, same AMP scaler usage -- and only adds the
        auxiliary output of the network and the per-term reporting.
        """
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Same autocast policy as nnU-Net: enabled on CUDA only (CPU autocast is
        # much slower and MPS refuses it even when disabled).
        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" \
                else dummy_context():
            output = self.network(data, return_aux=True)
            loss, components = self.loss(output, target, return_components=True)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        reported = {"loss": loss.detach().cpu().numpy()}
        for name, value in components.items():
            reported[f"loss_{name}"] = float(value.detach())
        for name, value in output["stats"].items():
            reported[name] = float(value)
        return reported

    def validation_step(self, batch: dict) -> dict:
        """Validation step.

        The reported loss is the same compound objective optimised during
        training, so ``val_losses`` stays comparable with ``train_losses``. The
        pseudo-Dice is computed from the *main* segmentation logits only: the
        auxiliary phase heads are a supervision device and must never be treated
        as the model's prediction.
        """
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" \
                else dummy_context():
            network_output = self.network(data, return_aux=True)
            loss = self.loss(network_output, target)

        output = network_output["seg"]
        del network_output
        del data

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float16
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # [1:] in order to remove the background Dice
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }

    def on_train_epoch_end(self, train_outputs: List[dict]) -> None:
        """Inherited loss logging plus VCAU-Net's per-term diagnostics.

        The alignment / reliability numbers are what tell you whether 3D-SPCA
        and UGCF are actually doing anything: ``mean_flow_magnitude`` collapsing
        to zero means the warp has become the identity, and the reliability gap
        between C2 and C5 says which phase the fusion is trusting.
        """
        super().on_train_epoch_end(train_outputs)
        if self.local_rank != 0:
            return

        outputs = collate_outputs(train_outputs)

        def _mean(key: str):
            return float(np.mean(outputs[key])) if key in outputs else None

        parts = []
        for name in ("main", "aux", "boundary", "align", "smooth"):
            value = _mean(f"loss_{name}")
            if value is not None:
                parts.append(f"{name}={value:.4f}")
        self.print_to_log_file("[VCAU-Net] train " + " ".join(parts))

        diagnostics = []
        for name, label in (
            ("mean_entropy_c2", "entropy_C2"),
            ("mean_entropy_c5", "entropy_C5"),
            ("mean_reliability_c2", "reliability_C2"),
            ("mean_reliability_c5", "reliability_C5"),
            ("mean_flow_magnitude", "flow|mean|"),
            ("max_flow_magnitude", "flow|max|"),
        ):
            value = _mean(name)
            if value is not None:
                diagnostics.append(f"{label}={value:.4f}")
        self.print_to_log_file("[VCAU-Net] " + " ".join(diagnostics))

    def _do_i_compile(self):
        # The training forward returns a structured dict and the method-specific
        # modules contain explicit Python loops (the selective scan), so eager
        # execution is both more predictable and no slower in practice.
        return False
