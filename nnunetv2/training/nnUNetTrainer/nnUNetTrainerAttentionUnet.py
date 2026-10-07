import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import PlansDrivenConfig, adapt_model_to_nnunet_base_chain
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class nnUNetTrainerAttentionUnet(nnUNetTrainer):
    """AttentionUnet on nnU-Net base chain. DS via logits pyramid fallback."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[AttentionUnet] DS via logits pyramid fallback (documented)")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        from monai.networks.nets import AttentionUnet

        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)

        channels = tuple(cfg.features_per_stage[:5]) if len(cfg.features_per_stage) >= 3 else (32, 64, 128, 256, 320)
        strides_list = [tuple(int(x) for x in s) for s in cfg.pool_op_kernel_sizes]
        strides = tuple(strides_list[:len(channels)-1]) if len(strides_list) >= len(channels)-1 else tuple((2,2,2) for _ in range(len(channels)-1))

        raw = AttentionUnet(
            spatial_dims=cfg.dim, in_channels=num_input_channels, out_channels=num_output_channels,
            channels=channels, strides=strides, kernel_size=3, up_kernel_size=3, dropout=0.0,
        )
        model = adapt_model_to_nnunet_base_chain(raw, cfg, model_name="AttentionUnet", fallback_ds=True)
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)
