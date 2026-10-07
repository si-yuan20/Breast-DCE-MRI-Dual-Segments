import math
from typing import Union, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.base_chain_adapters import (
    PlansDrivenConfig,
    adapt_model_to_nnunet_base_chain,
)
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import propagate_deep_supervision_toggle


class _UNETRWithDS(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, spatial_dims=3):
        super().__init__()
        from monai.networks.nets import UNETR
        self.unetr = UNETR(in_channels=in_channels, out_channels=out_channels, img_size=img_size,
                           feature_size=feature_size, hidden_size=hidden_size, mlp_dim=mlp_dim,
                           num_heads=num_heads, proj_type="conv", norm_name="instance",
                           res_block=True, dropout_rate=0.0, spatial_dims=spatial_dims,
                           qkv_bias=False, save_attn=False)
        self.img_size = tuple(int(i) for i in img_size)

    @staticmethod
    def _crop_or_pad(x, target):
        if tuple(x.shape[2:]) == target: return x
        cur = tuple(x.shape[2:])
        slices = [slice(None), slice(None)]
        for cs, ts in zip(cur, target):
            if cs > ts: st = (cs-ts)//2; slices.append(slice(st, st+ts))
            else: slices.append(slice(None))
        x = x[tuple(slices)]
        pads = []
        for cs, ts in zip(reversed(tuple(x.shape[2:])), reversed(target)):
            if cs < ts: diff = ts-cs; pads.extend([diff//2, diff-diff//2])
            else: pads.extend([0,0])
        if any(pads): x = F.pad(x, pads, mode="reflect")
        return x

    def forward(self, x):
        orig = tuple(x.shape[2:])
        if orig != self.img_size: x = self._crop_or_pad(x, self.img_size)
        out = self.unetr(x)
        if tuple(out.shape[2:]) != orig: out = self._crop_or_pad(out, orig)
        return out


class nnUNetTrainerUNETR(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.print_to_log_file("[UNETR] DS via logits pyramid fallback (documented)")

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                    arch_init_kwargs_req_import, num_input_channels,
                                    num_output_channels, enable_deep_supervision=True):
        cfg = PlansDrivenConfig.from_nnunet(self.plans_manager, self.dataset_json, self.configuration_manager, num_input_channels)
        patch = cfg.patch_size
        img_size = tuple(int(math.ceil(s/16)*16) for s in patch)
        raw = _UNETRWithDS(in_channels=num_input_channels, out_channels=num_output_channels, img_size=img_size)
        model = adapt_model_to_nnunet_base_chain(raw, cfg, model_name="UNETR", fallback_ds=True)
        return model

    def set_deep_supervision_enabled(self, enabled):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)
