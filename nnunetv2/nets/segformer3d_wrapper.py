# nnunetv2/net/segformer3d_wrapper.py
import torch.nn as nn
from .segformer3d import build_segformer3d_model

class SegFormer3D_Wrapper(nn.Module):
    """
    nnUNet wants forward() -> tuple(outputs) when deep-supervision is used.
    We keep things simple and return a 1-tuple.
    """
    def __init__(self,
                 in_channels: int,
                 n_classes: int,
                 cfg_overrides: dict | None = None,
                 deep_supervision: bool = False):
        super().__init__()
        cfg = {
            "model_parameters": {
                "in_channels": in_channels,
                "num_classes": n_classes,
                # --- defaults taken from the original file ---
                "sr_ratios": [4, 2, 1, 1],
                "embed_dims": [32, 64, 160, 256],
                "patch_kernel_size": [7, 3, 3, 3],
                "patch_stride": [4, 2, 2, 2],
                "patch_padding": [3, 1, 1, 1],
                "mlp_ratios": [4, 4, 4, 4],
                "num_heads": [1, 2, 5, 8],
                "depths": [2, 2, 2, 2],
                "decoder_head_embedding_dim": 256,
                "decoder_dropout": 0.0,
            }
        }
        if cfg_overrides is not None:
            cfg["model_parameters"].update(cfg_overrides)

        self.deep_supervision = deep_supervision
        self.model = build_segformer3d_model(cfg)

    def forward(self, x):
        out = self.model(x)                # (B, C, D, H, W)
        if self.deep_supervision:
            return (out,)                  # 1-tuple for nnUNet¡¯s DS loss
        return out