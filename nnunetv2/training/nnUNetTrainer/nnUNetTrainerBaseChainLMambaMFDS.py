import torch
from torch import nn
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import PlansDrivenConfig, adapt_model_to_nnunet_base_chain
from nnunetv2.nets.base_chain_lmamba_mfds import BaseChainLMambaMFDS
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class nnUNetTrainerBaseChainLMambaMFDS(nnUNetTrainer):
    """
    PROPOSED METHOD — nnU-Net backbone + LMamba/MFDS/CSDF plugins.
    DS: REAL (nnU-Net decoder DS heads).
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda"),
                 **kwargs):
        # Plugin config — read from kwargs to avoid polluting base __init__ signature
        self._use_lmamba = kwargs.pop('use_lmamba_bottleneck', True)
        self._use_mfds = kwargs.pop('use_mfds_skip', True)
        self._use_csdf = kwargs.pop('use_csdf_decoder', True)
        self._compress = kwargs.pop('lmamba_compress_ratio', 2)
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.enable_deep_supervision = True
        self.print_to_log_file(
            f"[Proposed] BaseChainLMambaMFDS. "
            f"Plugins: LMamba={self._use_lmamba} MFDS={self._use_mfds} CSDF={self._use_csdf}"
        )

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)
        dim = cfg.dim
        conv_op = convert_dim_to_conv_op(dim)
        instnorm = get_matching_instancenorm(dimension=dim)

        raw = BaseChainLMambaMFDS(
            input_channels=num_input_channels, num_classes=num_output_channels,
            n_stages=cfg.n_stages, features_per_stage=cfg.features_per_stage,
            conv_op=conv_op, kernel_sizes=cfg.conv_kernel_sizes,
            strides=cfg.pool_op_kernel_sizes,
            n_conv_per_stage=cfg.n_conv_per_stage_encoder,
            n_conv_per_stage_decoder=cfg.n_conv_per_stage_decoder,
            conv_bias=True, norm_op=instnorm,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None, dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU, nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
            deep_supervision=True, pool_type='conv',
            use_lmamba_bottleneck=self._use_lmamba,
            use_mfds_skip=self._use_mfds,
            use_csdf_decoder=self._use_csdf,
            lmamba_compress_ratio=self._compress,
            mfds_gamma_init=0.1,
        )
        raw.apply(InitWeights_He(neg_slope=1e-2))

        model = adapt_model_to_nnunet_base_chain(
            raw, cfg, model_name="BaseChainLMambaMFDS", has_internal_ds=True,
        )
        self.print_to_log_file(f"[Proposed] Built. features={cfg.features_per_stage}")
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)
        # Safety net: also walk through the wrapper chain manually and
        # toggle deep_supervision on EVERY layer that has it (skipping
        # size_divisibility-carrying wrappers like ShapeStrictWrapper3D
        # which don't have meaningful DS flags).  This ensures the
        # BaseChainFeatureAdapter3D wrapper gets toggled alongside the
        # inner BaseChainLMambaMFDS model.
        net = self.network
        if net is None: return
        if hasattr(net, "module"): net = net.module
        if hasattr(net, "_orig_mod"): net = net._orig_mod
        while True:
            if hasattr(net, "deep_supervision") and not hasattr(net, "size_divisibility"):
                net.deep_supervision = bool(enabled)
            if hasattr(net, "model"):
                net = net.model
            else:
                break

    def _do_i_compile(self):
        self.print_to_log_file("[Compile] disabled — mamba_ssm incompatible")
        return False


# Ablation subclasses
class nnUNetTrainerBaseChainLMambaMFDS_NoMamba(nnUNetTrainerBaseChainLMambaMFDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_lmamba_bottleneck=False, **kwargs)

class nnUNetTrainerBaseChainLMambaMFDS_NoMFDS(nnUNetTrainerBaseChainLMambaMFDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_mfds_skip=False, **kwargs)

class nnUNetTrainerBaseChainLMambaMFDS_NoCSDF(nnUNetTrainerBaseChainLMambaMFDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_csdf_decoder=False, **kwargs)

class nnUNetTrainerBaseChainLMambaMFDS_NoPlugins(nnUNetTrainerBaseChainLMambaMFDS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, use_lmamba_bottleneck=False, use_mfds_skip=False, use_csdf_decoder=False, **kwargs)
