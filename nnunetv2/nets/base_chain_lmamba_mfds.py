"""
BaseChainLMambaMFDS — nnU-Net Backbone + Mamba/MFDS/CSDF Plugins
==================================================================
Proposed method for SCI paper.

Design principle: Keep the nnU-Net encoder/decoder architecture FULLY INTACT.
All plans-derived parameters (features_per_stage, strides, kernel_sizes,
norm, nonlin, etc.) are automatically inherited.

Three lightweight plugins are inserted at specific points:
  1. LMambaBlock3D → replaces part of the bottleneck conv stack
  2. MFDSBlock3D   → augments each skip connection path
  3. CSDFBlock3D   → replaces the decoder's simple concat+conv fusion

All plugins are independently toggleable for ablation studies.
"""

from typing import Union, List, Tuple, Type, Sequence
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import (
    get_matching_pool_op,
    get_matching_convtransp,
    maybe_convert_scalar_to_list,
    convert_conv_op_to_dim,
)
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import (
    StackedResidualBlocks,
    BasicBlockD,
)
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import (
    init_last_bn_before_add_to_0,
)

from nnunetv2.nets.lmamba_mfds_unet import (
    LMambaBlock3D,
    MFDSBlock3D,
    CSDFBlock3D,
    ConvNormAct3D,
    BasicBlock3D,
    SafeGroupNorm,
    match_spatial_shape,
)
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import init_gate_bias_to_positive


class _PluginIdentity(nn.Module):
    """Placeholder for disabled plugins — zero overhead."""
    def forward(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            return args[0]
        return args[0] if args else None


class BaseChainLMambaMFDS(nn.Module):
    """
    nnU-Net backbone + Mamba/MFDS/CSDF plugins.

    Plugin insertion points:
      - Bottleneck: after encoder, before decoder. LMambaBlock3D optionally
        augments (or replaces) the deepest conv stage.
      - Skip connections: MFDSBlock3D augments each encoder skip before
        concatenation into the decoder. This applies frequency-domain
        attention and dual-path fusion on the skip features.
      - Decoder fusion: CSDFBlock3D replaces the naive concat+conv with
        a learnable gate that balances encoder and decoder contributions.

    Parameters
    ----------
    All standard nnU-Net params (from plans):
        input_channels, num_classes, n_stages, features_per_stage,
        conv_op, kernel_sizes, strides, n_conv_per_stage,
        n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
        dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
        deep_supervision, pool_type

    Plugin switches:
        use_lmamba_bottleneck : bool
        use_mfds_skip         : bool
        use_csdf_decoder      : bool
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        pool_type: str = 'conv',
        # ── Plugin switches ──
        use_lmamba_bottleneck: bool = True,
        use_mfds_skip: bool = True,
        use_csdf_decoder: bool = True,
        # ── Plugin config ──
        lmamba_compress_ratio: int = 2,
        mfds_gamma_init: float = 0.1,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.features_per_stage = features_per_stage
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_bias = conv_bias

        # Normalise list args
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        self.n_stages = n_stages
        self.output_channels = features_per_stage

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # ── 1. Stem ──
        stem_channels = features_per_stage[0]
        self.stem = StackedConvBlocks(
            1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1,
            conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs,
        )

        # ── 2. Encoder stages ──
        self.encoder_stages = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        encoder_input_channels = stem_channels

        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1

            stage = StackedResidualBlocks(
                n_conv_per_stage[s], conv_op, encoder_input_channels,
                features_per_stage[s], kernel_sizes[s], stride_for_conv,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                nonlin, nonlin_kwargs, block=BasicBlockD,
            )

            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            self.encoder_stages.append(stage)
            encoder_input_channels = features_per_stage[s]

        # ── 3. Bottleneck plugins ──
        bottleneck_ch = features_per_stage[-1]
        if use_lmamba_bottleneck:
            self.lmamba_bottleneck = nn.Sequential(
                LMambaBlock3D(bottleneck_ch, compress_ratio=lmamba_compress_ratio),
                BasicBlock3D(bottleneck_ch, bottleneck_ch),
            )
        else:
            self.lmamba_bottleneck = _PluginIdentity()

        # ── 4. Skip connection plugins ──
        skip_channels = features_per_stage[:-1]
        if use_mfds_skip:
            self.mfds_skips = nn.ModuleList([
                MFDSBlock3D(ch) for ch in skip_channels
            ])
            # Override gamma init for MFDS blocks
            for mfds in self.mfds_skips:
                mfds.gamma = nn.Parameter(torch.tensor(mfds_gamma_init))
        else:
            self.mfds_skips = nn.ModuleList([_PluginIdentity() for _ in skip_channels])

        # ── 5. Decoder ──
        transpconv_op = get_matching_convtransp(conv_op=conv_op)
        self.decoder_stages = nn.ModuleList()
        self.decoder_transpconvs = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        if use_csdf_decoder:
            self.csdf_blocks = nn.ModuleList()
        else:
            self.csdf_blocks = None

        for s in range(1, n_stages):
            input_features_below = features_per_stage[-s]
            input_features_skip = features_per_stage[-(s + 1)]
            stride_for_transpconv = self.strides[-s]

            # Transposed conv for upsampling
            transpconv = transpconv_op(
                input_features_below, input_features_skip,
                stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias,
            )
            self.decoder_transpconvs.append(transpconv)

            if use_csdf_decoder:
                # CSDF gate fusion (replaces concat+conv)
                csdf = CSDFBlock3D(
                    enc_ch=input_features_skip,
                    dec_ch=input_features_skip,
                    out_ch=input_features_skip,
                )
                self.csdf_blocks.append(csdf)
            else:
                # Standard concat+conv
                self.decoder_stages.append(
                    StackedResidualBlocks(
                        n_blocks=n_conv_per_stage_decoder[s - 1],
                        conv_op=conv_op,
                        input_channels=2 * input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=kernel_sizes[-(s + 1)],
                        initial_stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        dropout_op=dropout_op,
                        dropout_op_kwargs=dropout_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    )
                )

            # Deep supervision head (always built, toggle via self.deep_supervision)
            self.seg_layers.append(conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        # ── Initialisation ──
        self.apply(init_last_bn_before_add_to_0)
        if use_csdf_decoder:
            init_gate_bias_to_positive(self, bias_value=2.0)

    def forward(self, x):
        # ── Encoder ──
        skips = []
        x = self.stem(x)
        skips.append(x)

        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)

        # ── Bottleneck LMamba ──
        x = self.lmamba_bottleneck(x)

        # ── Decoder with skip plugins ──
        seg_outputs = []
        for s in range(len(self.decoder_transpconvs)):
            # Upsample
            x_up = self.decoder_transpconvs[s](x)
            skip = skips[-(s + 2)]

            # MFDS skip augmentation (mfds_skips is shallow→deep, decoder
            # accesses skips from the deep end, so reverse the index)
            skip = self.mfds_skips[-(s + 1)](skip)

            # Match spatial shapes
            x_up = match_spatial_shape(x_up, skip)

            if self.csdf_blocks is not None:
                # CSDF gate fusion
                x = self.csdf_blocks[s](skip, x_up)
            else:
                # Standard concat + conv
                x = torch.cat((x_up, skip), dim=1)
                x = self.decoder_stages[s](x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.decoder_transpconvs) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

        # Reverse: full-resolution output first (nnU-Net convention)
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[0]
        return seg_outputs

    def compute_conv_feature_map_size(self, input_size):
        """Estimate total feature map size for memory planning."""
        output = np.int64(0)
        # stem
        output += self.stem.compute_conv_feature_map_size(input_size)
        # encoder
        for s in range(self.n_stages):
            output += self.encoder_stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output
