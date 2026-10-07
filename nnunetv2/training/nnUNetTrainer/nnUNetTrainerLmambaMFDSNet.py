import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import PlansDrivenConfig, adapt_model_to_nnunet_base_chain
from nnunetv2.nets.lmamba_mfds_unet import LMambaMFDSUNet3D
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class nnUNetTrainerLMambaMFDSNet(nnUNetTrainer):
    """LMambaMFDSNet on nnU-Net base chain. DS: REAL (from decoder features)."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[LMambaMFDSNet] DS via decoder features (REAL).")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)

        base_channels = cfg.features_per_stage[0] if cfg.features_per_stage else 32
        base_channels = max(16, int(base_channels))
        max_channels = cfg.unet_max_num_features

        # pool_strides from configuration_manager
        pool_strides = [(1,2,2), (2,2,2), (2,2,2), (2,2,2)]
        if cfg.pool_op_kernel_sizes:
            raw_strides = [tuple(int(x) for x in s) for s in cfg.pool_op_kernel_sizes]
            raw_strides = [s for s in raw_strides if s != (1,1,1)]
            if len(raw_strides) >= 2:
                pool_strides = raw_strides

        raw = LMambaMFDSUNet3D(
            input_channels=num_input_channels, num_classes=num_output_channels,
            base_channels=base_channels, pool_strides=pool_strides,
            deep_supervision=True, max_channels=max_channels,
        )
        model = adapt_model_to_nnunet_base_chain(
            raw, cfg, model_name="LMambaMFDSNet", has_internal_ds=True,
        )
        self.print_to_log_file(
            f"[LMambaMFDSNet] Built. base_ch={base_channels} max_ch={max_channels} strides={pool_strides}"
        )
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)
        net = self.network
        if net is None: return
        if hasattr(net, "module"): net = net.module
        if hasattr(net, "_orig_mod"): net = net._orig_mod
        if hasattr(net, "model") and hasattr(net.model, "model") and hasattr(net.model.model, "set_deep_supervision"):
            net.model.model.set_deep_supervision(enabled)

    def _do_i_compile(self):
        self.print_to_log_file("[Compile] disabled — mamba_ssm incompatible")
        return False
