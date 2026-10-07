import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import PlansDrivenConfig, adapt_model_to_nnunet_base_chain
from nnunetv2.nets.segformer3d import SegFormer3D
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class nnUNetTrainerSegFormer(nnUNetTrainer):
    """SegFormer3D on nnU-Net base chain. DS via logits pyramid fallback."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[SegFormer3D] DS via logits pyramid fallback (documented)")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)

        raw = SegFormer3D(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
        )
        model = adapt_model_to_nnunet_base_chain(raw, cfg, model_name="SegFormer3D", fallback_ds=True)
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)

    def _do_i_compile(self):
        """SegFormer3D uses spatial reshape from token count (cube_root)
        which is incompatible with torch.compile tracing on anisotropic patches."""
        self.print_to_log_file("[Compile] disabled — SegFormer3D spatial reshape incompatible with torch.compile")
        return False
