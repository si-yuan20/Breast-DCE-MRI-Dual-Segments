import math
from typing import Union
import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import PlansDrivenConfig, adapt_model_to_nnunet_base_chain
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class _SwinUNETRDS(nn.Module):
    """MONAI SwinUNETR → reflect-pad wrapper → DS fallback ready."""

    def __init__(self, in_channels, out_channels, img_size, feature_size=48,
                 depths=(2,2,2,2), num_heads=(3,6,12,24), spatial_dims=3):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        self.img_size = tuple(int(i) for i in img_size)
        try:
            self.swin = SwinUNETR(
                img_size=img_size, in_channels=in_channels, out_channels=out_channels,
                depths=depths, num_heads=num_heads, feature_size=feature_size,
                norm_name="instance", drop_rate=0.0, attn_drop_rate=0.0,
                dropout_path_rate=0.0, normalize=True, use_checkpoint=False,
                spatial_dims=spatial_dims, downsample="merging", use_v2=False,
            )
        except TypeError:
            self.swin = SwinUNETR(
                in_channels=in_channels, out_channels=out_channels,
                depths=depths, num_heads=num_heads, feature_size=feature_size,
                norm_name="instance", drop_rate=0.0, attn_drop_rate=0.0,
                dropout_path_rate=0.0, normalize=True, use_checkpoint=False,
                spatial_dims=spatial_dims, downsample="merging", use_v2=False,
            )

    @staticmethod
    def _reflect_pad_crop(x, target):
        if tuple(x.shape[2:]) == target: return x
        cur = tuple(x.shape[2:])
        slices = [slice(None), slice(None)]
        for cs, ts in zip(cur, target):
            if cs > ts: st = (cs - ts) // 2; slices.append(slice(st, st + ts))
            else: slices.append(slice(None))
        x = x[tuple(slices)]
        pads = []
        for cs, ts in zip(reversed(tuple(x.shape[2:])), reversed(target)):
            if cs < ts: diff = ts - cs; pads.extend([diff//2, diff-diff//2])
            else: pads.extend([0,0])
        if any(pads): x = F.pad(x, pads, mode="reflect")
        return x

    def forward(self, x):
        orig = tuple(x.shape[2:])
        out = self.swin(self._reflect_pad_crop(x, self.img_size))
        if isinstance(out, (list, tuple)): out = out[0]
        return self._reflect_pad_crop(out, orig)


class nnUNetTrainerSwinUNETR(nnUNetTrainer):
    """SwinUNETR on nnU-Net base chain. DS via logits pyramid fallback."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[SwinUNETR] DS via logits pyramid fallback (documented)")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        cfg = PlansDrivenConfig.from_nnunet(
            self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)
        divisor = 2**5
        img_size = tuple(int(math.ceil(s/divisor)*divisor) for s in cfg.patch_size)

        raw = _SwinUNETRDS(in_channels=num_input_channels, out_channels=num_output_channels, img_size=img_size)
        model = adapt_model_to_nnunet_base_chain(raw, cfg, model_name="SwinUNETR", fallback_ds=True)
        self.print_to_log_file(f"[SwinUNETR] Built. img_size={img_size}")
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)
