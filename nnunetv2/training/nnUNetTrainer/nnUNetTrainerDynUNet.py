import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import PlansDrivenConfig, adapt_model_to_nnunet_base_chain
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class nnUNetTrainerDynUNet(nnUNetTrainer):
    """DynUNet on nnU-Net base chain. DS via logits pyramid fallback."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[DynUNet] DS via logits pyramid fallback (documented)")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        from monai.networks.nets import DynUNet

        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)

        kernel_size = [tuple(int(x) for x in s) for s in cfg.conv_kernel_sizes]
        strides = [tuple(int(x) for x in s) for s in cfg.pool_op_kernel_sizes]
        filters = cfg.features_per_stage
        upsample_kernel_size = strides[1:] if len(strides) > 1 else [(2,2,2)]

        raw = DynUNet(
            spatial_dims=cfg.dim, in_channels=num_input_channels, out_channels=num_output_channels,
            kernel_size=kernel_size, strides=strides, upsample_kernel_size=upsample_kernel_size,
            filters=filters, dropout=0.0, norm_name="instance",
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=False, deep_supr_num=1, res_block=True,
        )
        model = adapt_model_to_nnunet_base_chain(raw, cfg, model_name="DynUNet", fallback_ds=True)
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)
