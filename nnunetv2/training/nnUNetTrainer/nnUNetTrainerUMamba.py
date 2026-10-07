import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import PlansDrivenConfig, adapt_model_to_nnunet_base_chain
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class nnUNetTrainerUMamba(nnUNetTrainer):
    """UMamba on nnU-Net base chain. DS: REAL (from UNetResDecoder)."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[UMamba] DS via UNetResDecoder (REAL).")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        from nnunetv2.nets.UMamba import get_umamba_enc_from_plans

        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)

        raw = get_umamba_enc_from_plans(
            self.plans_manager, self.dataset_json, self.configuration_manager,
            num_input_channels, deep_supervision=True,
        )

        model = adapt_model_to_nnunet_base_chain(
            raw, cfg, model_name="UMamba", has_internal_ds=True,
        )
        self.print_to_log_file("[UMamba] Built with real DS.")
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)

    def _do_i_compile(self):
        self.print_to_log_file("[Compile] disabled — mamba_ssm incompatible")
        return False
