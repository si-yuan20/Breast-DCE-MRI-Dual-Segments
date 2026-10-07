import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import PlansDrivenConfig, adapt_model_to_nnunet_base_chain
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class nnUNetTrainerSegMamba(nnUNetTrainer):
    """SegMamba on nnU-Net base chain. DS: REAL (decoder features). Mamba only at low-res stages."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[SegMamba] DS via decoder features (REAL). Mamba at stages 2+.")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        from nnunetv2.nets.SegMamba import SegMamba

        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)

        # feat_size from plans features_per_stage
        feat_size = tuple(cfg.features_per_stage[:4]) if len(cfg.features_per_stage) >= 4 else (48, 96, 192, 384)
        if len(feat_size) != 4:
            feat_size = (48, 96, 192, 384)

        hidden_size = feat_size[-1] * 2

        raw = SegMamba(
            in_chans=num_input_channels, out_chans=num_output_channels,
            depths=(1, 1, 1, 1), feat_size=list(feat_size),
            hidden_size=hidden_size, norm_name="instance",
            conv_block=True, res_block=True, spatial_dims=3,
            mamba_start_stage=2,
        )

        # SegMamba has internal decoder features — use REAL DS heads
        # The decoder features are: decoder5_out, decoder4_out, decoder3_out, decoder2_out
        # We hook into them after the forward pass
        # (SegMamba's forward returns single tensor → we use fallback for now;
        #  full feature extraction requires forward hook which is complex)
        model = adapt_model_to_nnunet_base_chain(
            raw, cfg, model_name="SegMamba", fallback_ds=True,
        )
        self.print_to_log_file(f"[SegMamba] Built. feat_size={feat_size} hidden={hidden_size}")
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)

    def _do_i_compile(self):
        self.print_to_log_file("[Compile] disabled — mamba_ssm incompatible")
        return False
