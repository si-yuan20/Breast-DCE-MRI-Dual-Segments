from typing import Union, List, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import (
    PlansDrivenConfig,
    adapt_model_to_nnunet_base_chain,
    NormSanitizer3D,
)
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


# ---------------------------------------------------------------------------
# SegResNet adapter — InstanceNorm + trilinear upsample
# ---------------------------------------------------------------------------

class _SegResNetForNNUNet(nn.Module):
    """MONAI SegResNet with InstanceNorm3d, no dropout, exact trilinear upsample."""

    def __init__(self, spatial_dims=3, init_filters=32, in_channels=1, out_channels=2):
        super().__init__()
        from monai.networks.nets import SegResNet
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels

        # Build SegResNet with InstanceNorm (try multiple MONAI API variants)
        base_kwargs = dict(
            spatial_dims=spatial_dims,
            init_filters=init_filters,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=0.0,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            upsample_mode="nontrainable",
        )

        built = False
        try:
            self._net = SegResNet(norm="instance", **base_kwargs)
            built = True
        except Exception:
            pass
        if not built:
            try:
                self._net = SegResNet(norm_name="INSTANCE", **base_kwargs)
                built = True
            except Exception:
                pass
        if not built:
            self._net = SegResNet(**base_kwargs)
            NormSanitizer3D.replace_bn(self._net)

    def forward(self, x):
        # Delegate to MONAI SegResNet's internal forward
        net = self._net
        x0 = net.convInit(x)
        if net.dropout is not None:
            x0 = net.dropout(x0)

        down_x = [x0]
        for i in range(len(net.blocks_down)):
            down_x.append(net.blocks_down[i](down_x[-1]))

        x = down_x[-1]
        for i, (up, upl) in enumerate(zip(net.up_samples, net.up_layers)):
            skip = down_x[-(i + 2)]
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = x + skip
            x = upl(x)

        if hasattr(net, "conv_final"):
            x = net.conv_final(x)
        return x


# ---------------------------------------------------------------------------
# nnUNetTrainerSegResNet (DS via logits pyramid fallback)
# ---------------------------------------------------------------------------

class nnUNetTrainerSegResNet(nnUNetTrainer):
    """
    SegResNet on nnU-Net base chain.

    - DS: ENABLED (logits pyramid fallback — documented)
    - Norm: InstanceNorm3d
    - BN: BANNED
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[SegResNet] DS via logits pyramid fallback (documented)")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json,
            self.configuration_manager, num_input_channels,
        )

        init_filters = cfg.features_per_stage[0] if cfg.features_per_stage else 32
        init_filters = max(16, int(init_filters))

        raw = _SegResNetForNNUNet(
            spatial_dims=3, init_filters=init_filters,
            in_channels=num_input_channels, out_channels=num_output_channels,
        )

        model = adapt_model_to_nnunet_base_chain(
            raw, cfg, model_name="SegResNet", fallback_ds=True,
        )
        self.print_to_log_file(f"[SegResNet] Built. {cfg.summary()}")
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)
