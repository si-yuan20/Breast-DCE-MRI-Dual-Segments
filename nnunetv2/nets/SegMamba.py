from __future__ import annotations

import inspect
import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba


# ---------------------------------------------------------------------------
# LayerNorm helper (channels_first support)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ("channels_last", "channels_first"):
            raise NotImplementedError("Unsupported data_format: {}".format(self.data_format))
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


# ---------------------------------------------------------------------------
# MambaLayer — with residual scaling (LayerScale)
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_slices: Union[int, None] = None,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim)

        kwargs = {
            "d_model": self.dim,
            "d_state": int(d_state),
            "d_conv": int(d_conv),
            "expand": int(expand),
        }

        try:
            params = inspect.signature(Mamba).parameters
            if "bimamba_type" in params:
                kwargs["bimamba_type"] = "v3"
            if "nslices" in params and num_slices is not None:
                kwargs["nslices"] = int(num_slices)
        except Exception:
            pass

        self.mamba = Mamba(**kwargs)

        # LayerScale — stabilises residual path
        self.layer_scale = nn.Parameter(torch.tensor(layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.detach()  # detach to avoid double counting in autograd

        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        batch_size, channels = x.shape[:2]
        if channels != self.dim:
            raise RuntimeError(
                "MambaLayer channel mismatch: expected {}, got {}.".format(self.dim, channels)
            )

        spatial_shape = x.shape[2:]
        num_tokens = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]

        x_flat = x.reshape(batch_size, channels, num_tokens).transpose(1, 2)
        x_flat = self.norm(x_flat)
        x_flat = self.mamba(x_flat)
        x = x_flat.transpose(1, 2).reshape(batch_size, channels, *spatial_shape)

        if x.dtype != residual.dtype:
            x = x.to(dtype=residual.dtype)

        # Scaled residual
        return residual + self.layer_scale * x


# ---------------------------------------------------------------------------
# GSC (Grouped Spatial Convolution) — with LayerScale for stability
# ---------------------------------------------------------------------------

class GSC(nn.Module):
    def __init__(self, in_channels: int, layer_scale_init: float = 1e-4):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channels)
        self.act = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        self.proj2 = nn.Conv3d(in_channels, in_channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.act2 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        self.proj3 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channels)
        self.act3 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        self.proj4 = nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channels)
        self.act4 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)

        # LayerScale for the residual
        self.layer_scale = nn.Parameter(torch.tensor(layer_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.act(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.act2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.act3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.act4(x)

        return residual + self.layer_scale * x


# ---------------------------------------------------------------------------
# MlpChannel
# ---------------------------------------------------------------------------

class MlpChannel(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, mlp_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# ConvBlock3D — pure convolutional block for high-resolution stages
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """nnU-Net-style residual conv block for stages where Mamba is inappropriate."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(channels)
        self.act1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, 1, kernel_size // 2, bias=False)

        self.norm2 = nn.InstanceNorm3d(channels)
        self.act2 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, 1, kernel_size // 2, bias=False)

    def forward(self, x):
        residual = x
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return residual + x


# ---------------------------------------------------------------------------
# MambaEncoder — Mamba ONLY in low-resolution stages
# ---------------------------------------------------------------------------

class MambaEncoder(nn.Module):
    """
    Mamba-based encoder with the following strategy:
      - Stage 0 & 1 (HIGH resolution): Pure Conv3d residual blocks.
        Mamba is NOT used at high resolution because token counts
        (100K–300K) are too long for the SSM scan.
      - Stage 2 & 3 (LOW resolution): Mamba blocks with LayerScale.
        Token counts (500–8K) are manageable.
    """

    def __init__(
        self,
        in_chans: int = 1,
        depths: Sequence[int] = (1, 1, 1, 1),
        dims: Sequence[int] = (48, 96, 192, 384),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        double_output_channels: bool = False,
        mamba_start_stage: int = 2,  # ★ Mamba only from stage 2 onward
    ):
        super().__init__()

        if len(depths) != 4:
            raise ValueError("depths must contain 4 values.")
        if len(dims) != 4:
            raise ValueError("dims must contain 4 values.")

        self.depths = tuple(int(i) for i in depths)
        self.dims = tuple(int(i) for i in dims)
        self.out_indices = tuple(int(i) for i in out_indices)
        self.double_output_channels = bool(double_output_channels)
        self.mamba_start_stage = max(1, int(mamba_start_stage))

        self.downsample_layers = nn.ModuleList()

        # Stem: Conv7×7 stride 2 + InstanceNorm3d + LeakyReLU
        stem = nn.Sequential(
            nn.Conv3d(in_chans, self.dims[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm3d(self.dims[0]),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(self.dims[i]),
                nn.LeakyReLU(negative_slope=1e-2, inplace=True),
                nn.Conv3d(self.dims[i], self.dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.gscs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.mlps = nn.ModuleList()

        num_slices_list = (64, 32, 16, 8)

        for i in range(4):
            self.gscs.append(GSC(self.dims[i], layer_scale_init=1e-4))

            if i >= self.mamba_start_stage:
                # ── LOW resolution: use Mamba ──
                stage = nn.Sequential(
                    *[
                        MambaLayer(
                            dim=self.dims[i],
                            num_slices=num_slices_list[i],
                            layer_scale_init=layer_scale_init_value,
                        )
                        for _ in range(self.depths[i])
                    ]
                )
                stage_type = "Mamba"
            else:
                # ── HIGH resolution: pure Conv ──
                stage = nn.Sequential(
                    *[ConvBlock3D(self.dims[i]) for _ in range(self.depths[i])]
                )
                stage_type = "Conv"

            self.stages.append(stage)

            norm_layer = nn.InstanceNorm3d(self.dims[i])
            self.add_module("norm{}".format(i), norm_layer)

            if self.double_output_channels:
                self.mlps.append(MlpChannel(self.dims[i], 2 * self.dims[i]))
            else:
                self.mlps.append(nn.Identity())

            # Log token counts
            print(
                "[SegMamba] Stage {} | dims={}  depths={}  type={}".format(
                    i, self.dims[i], self.depths[i], stage_type
                )
            )

    @property
    def output_channels(self) -> Tuple[int, int, int, int]:
        if self.double_output_channels:
            return tuple(2 * i for i in self.dims)
        return self.dims

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            spatial = x.shape[2:]
            tokens = spatial[0] * spatial[1] * spatial[2]
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, "norm{}".format(i))
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outputs.append(x_out)

        return tuple(outputs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.forward_features(x)


# ---------------------------------------------------------------------------
# SegMamba — full network
# ---------------------------------------------------------------------------

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 13,
        depths: Sequence[int] = (2, 2, 2, 2),
        feat_size: Sequence[int] = (48, 96, 192, 384),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        hidden_size: int = 768,
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims: int = 3,
        double_encoder_output_channels: bool = False,
        mamba_start_stage: int = 2,
    ):
        super().__init__()

        if spatial_dims != 3:
            raise RuntimeError("SegMamba only supports 3D inputs.")
        if len(feat_size) != 4:
            raise ValueError("feat_size must contain 4 values.")
        if len(depths) != 4:
            raise ValueError("depths must contain 4 values.")

        self.hidden_size = int(hidden_size)
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.depths = tuple(int(i) for i in depths)
        self.feat_size = tuple(int(i) for i in feat_size)
        self.spatial_dims = int(spatial_dims)
        self.mamba_start_stage = int(mamba_start_stage)

        self.vit = MambaEncoder(
            in_chans=self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            double_output_channels=double_encoder_output_channels,
            mamba_start_stage=self.mamba_start_stage,
        )

        vit_channels = self.vit.output_channels

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims, in_channels=self.in_chans,
            out_channels=self.feat_size[0], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims, in_channels=vit_channels[0],
            out_channels=self.feat_size[1], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims, in_channels=vit_channels[1],
            out_channels=self.feat_size[2], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims, in_channels=vit_channels[2],
            out_channels=self.feat_size[3], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims, in_channels=vit_channels[3],
            out_channels=self.hidden_size, kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=self.spatial_dims, in_channels=self.hidden_size,
            out_channels=self.feat_size[3], kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=self.spatial_dims, in_channels=self.feat_size[3],
            out_channels=self.feat_size[2], kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=self.spatial_dims, in_channels=self.feat_size[2],
            out_channels=self.feat_size[1], kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=self.spatial_dims, in_channels=self.feat_size[1],
            out_channels=self.feat_size[0], kernel_size=3, upsample_kernel_size=2,
            norm_name=norm_name, res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims, in_channels=self.feat_size[0],
            out_channels=self.feat_size[0], kernel_size=3, stride=1,
            norm_name=norm_name, res_block=res_block,
        )
        self.out = UnetOutBlock(
            spatial_dims=self.spatial_dims, in_channels=self.feat_size[0],
            out_channels=self.out_chans,
        )

    @staticmethod
    def _match_spatial_shape(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        target_shape = tuple(ref.shape[2:])
        if tuple(x.shape[2:]) == target_shape:
            return x
        current_shape = tuple(x.shape[2:])
        slices = [slice(None), slice(None)]
        for current_size, target_size in zip(current_shape, target_shape):
            if current_size > target_size:
                start = (current_size - target_size) // 2
                end = start + target_size
                slices.append(slice(start, end))
            else:
                slices.append(slice(None))
        x = x[tuple(slices)]
        current_shape = tuple(x.shape[2:])
        pads = []
        need_pad = False
        for current_size, target_size in zip(reversed(current_shape), reversed(target_shape)):
            if current_size < target_size:
                diff = target_size - current_size
                left = diff // 2
                right = diff - left
                pads.extend([left, right])
                need_pad = True
            else:
                pads.extend([0, 0])
        if need_pad:
            x = F.pad(x, pads)
        return x

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        outs = self.vit(x_in)

        if len(outs) != 4:
            raise RuntimeError("SegMamba expected 4 encoder outputs, got {}.".format(len(outs)))

        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        x5 = outs[3]
        enc_hidden = self.encoder5(x5)

        dec3 = self.decoder5(enc_hidden, enc4)
        dec3 = self._match_spatial_shape(dec3, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec2 = self._match_spatial_shape(dec2, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec1 = self._match_spatial_shape(dec1, enc2)
        dec0 = self.decoder2(dec1, enc1)
        dec0 = self._match_spatial_shape(dec0, enc1)
        out = self.decoder1(dec0)

        return self.out(out)
