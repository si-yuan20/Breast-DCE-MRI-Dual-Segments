import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import PlansDrivenConfig, adapt_model_to_nnunet_base_chain
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class nnUNetTrainerVeloxSeg(nnUNetTrainer):
    """VeloxSeg on nnU-Net base chain (baseline only — SDKT disabled). DS via logits pyramid fallback."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[VeloxSeg] DS via logits pyramid fallback (documented). SDKT disabled — baseline only.")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        from nnunetv2.nets.VeloxSeg import VeloxSeg

        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)

        in_ch = [1 for _ in range(num_input_channels)]  # treat channels as "modalities"
        raw = VeloxSeg(
            input_size=cfg.patch_size, patch_size=4, in_ch=in_ch,
            n_classes=num_output_channels, deep_supervision=False, spatial_dim=3,
        )
        model = adapt_model_to_nnunet_base_chain(raw, cfg, model_name="VeloxSeg", fallback_ds=True)
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)
