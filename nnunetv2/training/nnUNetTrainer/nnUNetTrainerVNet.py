import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import PlansDrivenConfig, adapt_model_to_nnunet_base_chain
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class nnUNetTrainerVNet(nnUNetTrainer):
    """VNet on nnU-Net base chain. DS via logits pyramid fallback."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[VNet] DS via logits pyramid fallback (documented)")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        from monai.networks.nets import VNet

        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)

        raw = VNet(
            in_channels=num_input_channels, out_channels=num_output_channels,
            act=("elu", {"inplace": True}), dropout_dim=3, bias=False,
        )
        model = adapt_model_to_nnunet_base_chain(raw, cfg, model_name="VNet", fallback_ds=True)
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)
