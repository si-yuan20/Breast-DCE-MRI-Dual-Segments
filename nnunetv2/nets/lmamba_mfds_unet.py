from typing import List, Tuple, Union, Sequence
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def _to_3tuple(x):
    if isinstance(x, int):
        return (x, x, x)
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    assert len(x) == 3, f"Expected 3D tuple/list, got {x}"
    return tuple(int(i) for i in x)


def match_spatial_shape(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if src.shape[2:] == ref.shape[2:]:
        return src

    # Construct a crop-slice tuple using explicit ints – avoids the
    # "non-tuple sequence for multidimensional indexing" warning that
    # occurs when you pass a list of slices to [].
    crop_slices: list = [slice(None), slice(None)]
    for s, r in zip(src.shape[2:], ref.shape[2:]):
        if s > r:
            st = (s - r) // 2
            crop_slices.append(slice(st, st + r))
        else:
            crop_slices.append(slice(None))
    src = src[tuple(crop_slices)]  # explicit tuple() — safe

    # pad when src is smaller than ref in any spatial dim
    pad_list = []
    for s, r in zip(reversed(src.shape[2:]), reversed(ref.shape[2:])):
        if s < r:
            diff = r - s
            pad_list.extend([diff // 2, diff - diff // 2])
        else:
            pad_list.extend([0, 0])

    return F.pad(src, pad_list) if any(pad_list) else src


class SafeGroupNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        while groups > 1 and channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x):
        return self.norm(x)


class ConvNormAct3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, _to_3tuple(k), _to_3tuple(s), _to_3tuple(p), groups=groups, bias=False),
            SafeGroupNorm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1, 1)):
        super().__init__()
        stride = _to_3tuple(stride)

        self.conv1 = ConvNormAct3D(in_ch, out_ch, 3, stride, 1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, 1, 1, bias=False),
            SafeGroupNorm(out_ch),
        )

        self.proj = (
            nn.Conv3d(in_ch, out_ch, 1, stride, 0, bias=False)
            if in_ch != out_ch or stride != (1, 1, 1)
            else nn.Identity()
        )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.proj(x))

    def zero_init_last_norm(self):
        """Zero-init the last normalisation layer so the residual branch
        starts identity-like (same philosophy as nnU-Net's
        init_last_bn_before_add_to_0)."""
        last_norm = self.conv2[-1] if isinstance(self.conv2, nn.Sequential) and len(self.conv2) > 0 else None
        if last_norm is not None:
            if hasattr(last_norm, "weight") and last_norm.weight is not None:
                nn.init.constant_(last_norm.weight, 0)
            elif hasattr(last_norm, "norm") and hasattr(last_norm.norm, "weight") and last_norm.norm.weight is not None:
                nn.init.constant_(last_norm.norm.weight, 0)


class MultiScaleDepthwiseConvBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.b1 = nn.Conv3d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.b2 = nn.Conv3d(channels, channels, (1, 3, 3), 1, (0, 1, 1), groups=channels, bias=False)
        self.b3 = nn.Conv3d(channels, channels, (3, 1, 1), 1, (1, 0, 0), groups=channels, bias=False)
        self.b4 = nn.Conv3d(channels, channels, 3, 1, 2, dilation=2, groups=channels, bias=False)

        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 4, channels, 1, bias=False),
            SafeGroupNorm(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return x + self.fuse(y)


class AxialContextBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.d_conv = nn.Conv3d(channels, channels, (5, 1, 1), 1, (2, 0, 0), groups=channels, bias=False)
        self.h_conv = nn.Conv3d(channels, channels, (1, 5, 1), 1, (0, 2, 0), groups=channels, bias=False)
        self.w_conv = nn.Conv3d(channels, channels, (1, 1, 5), 1, (0, 0, 2), groups=channels, bias=False)

        self.proj = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            SafeGroupNorm(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        y = self.d_conv(x) + self.h_conv(x) + self.w_conv(x)
        return x + self.proj(y)


class MultiFrequencyDCTAttention3D(nn.Module):
    """Multi-axis 1D-DCT channel attention — true 3D frequency awareness.

    Instead of collapsing the depth dimension (``x.mean(dim=2)``) and
    computing a 2D DCT on the H-W plane only, this module:
      1. projects the feature volume onto each of the three spatial axes
         (D, H, W) via adaptive average pooling,
      2. computes **1D DCT coefficients** along each axis independently,
      3. aggregates per-axis statistics (mean / max),
      4. fuses them through a lightweight MLP to produce per-channel
         attention weights in (0, 1).

    This preserves depth-axis frequency structure for anisotropic 3D
    medical images while keeping the computational cost at O(C · B)
    — far cheaper than a full 3D DCT (O(C · B · D · H · W)).
    """

    def __init__(self, channels: int, freq_bins: int = 8, reduction: int = 8):
        super().__init__()
        self.freq_bins = int(freq_bins)

        # ---- 1D DCT basis (shared across D / H / W axes) ----
        dct1d = self._build_1d_dct_basis(self.freq_bins)          # (Bins, Bins)
        self.register_buffer("dct_basis_d", dct1d, persistent=False)
        self.register_buffer("dct_basis_h", dct1d.clone(), persistent=False)
        self.register_buffer("dct_basis_w", dct1d.clone(), persistent=False)

        # ---- lightweight MLP: 2×C  ->  C//reduction  ->  C  ----
        hidden = max(int(channels) // int(reduction), 4)
        self.mlp = nn.Sequential(
            nn.Conv3d(int(channels) * 2, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, int(channels), 1),
        )

    # ------------------------------------------------------------------
    # 1D DCT-II basis (orthonormal)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_1d_dct_basis(N: int) -> torch.Tensor:
        n = torch.arange(N).float()
        basis = []
        for k in range(N):
            alpha = math.sqrt(1.0 / N) if k == 0 else math.sqrt(2.0 / N)
            basis.append(alpha * torch.cos(math.pi * k * (n + 0.5) / N))
        return torch.stack(basis, dim=0)                          # (N, N)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape

        # --  project onto each axis  ----------------------------------
        # Depth axis
        proj_d = F.adaptive_avg_pool3d(x, (self.freq_bins, 1, 1))
        proj_d = proj_d[:, :, :, 0, 0]                            # (B, C, Bins)
        resp_d = torch.einsum("bcn,fn->bcf", proj_d, self.dct_basis_d)

        # Height axis
        proj_h = F.adaptive_avg_pool3d(x, (1, self.freq_bins, 1))
        proj_h = proj_h[:, :, 0, :, 0]                            # (B, C, Bins)
        resp_h = torch.einsum("bcn,fn->bcf", proj_h, self.dct_basis_h)

        # Width axis
        proj_w = F.adaptive_avg_pool3d(x, (1, 1, self.freq_bins))
        proj_w = proj_w[:, :, 0, 0, :]                            # (B, C, Bins)
        resp_w = torch.einsum("bcn,fn->bcf", proj_w, self.dct_basis_w)

        # --  aggregate statistics ------------------------------------
        avg_d, avg_h, avg_w = resp_d.mean(-1), resp_h.mean(-1), resp_w.mean(-1)
        mx_d,  mx_h,  mx_w  = (resp_d.amax(-1), resp_h.amax(-1), resp_w.amax(-1))

        # fuse across axes (simple average — keeps MLP compact)
        avg_fused = ((avg_d + avg_h + avg_w) / 3.0).view(B, C, 1, 1, 1)
        mx_fused  = ((mx_d  + mx_h  + mx_w)  / 3.0).view(B, C, 1, 1, 1)

        att = torch.cat([avg_fused, mx_fused], dim=1)             # (B, 2C, 1, 1, 1)
        return torch.sigmoid(self.mlp(att))


class MFDSBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.low_smooth = nn.Sequential(
            nn.Conv3d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            SafeGroupNorm(channels),
            nn.SiLU(inplace=True),
        )

        self.low_proj = ConvNormAct3D(channels, channels, 1, 1, 0)
        self.high_proj = ConvNormAct3D(channels, channels, 1, 1, 0)

        self.dct_att = MultiFrequencyDCTAttention3D(channels)
        self.weight_gen = nn.Conv3d(channels * 2, 2, 1)
        self.out_proj = nn.Conv3d(channels, channels, 1, bias=False)

        self.gamma = nn.Parameter(torch.tensor(0.1))  # ★ 1e-3 → 0.1: MFDS contributes from epoch 1

    def forward(self, x):
        x_low = F.avg_pool3d(x, 3, 1, 1)
        x_low = self.low_smooth(x_low)
        x_high = x - x_low

        f_low = self.low_proj(x_low)
        f_high = self.high_proj(x_high)

        a = self.dct_att(x)
        f_low = f_low * a
        f_high = f_high * a

        pooled = torch.cat(
            [
                F.adaptive_avg_pool3d(f_low, 1),
                F.adaptive_avg_pool3d(f_high, 1),
            ],
            dim=1,
        )

        w = torch.softmax(self.weight_gen(pooled), dim=1)
        fused = w[:, 0:1] * f_low + w[:, 1:2] * f_high

        return x + self.gamma.to(dtype=x.dtype) * self.out_proj(fused)


class MSFAEncoderBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride=(1, 1, 1), use_axial: bool = False):
        super().__init__()

        self.down = BasicBlock3D(in_ch, out_ch, stride=stride)
        self.ms = MultiScaleDepthwiseConvBlock3D(out_ch)
        self.axial = AxialContextBlock3D(out_ch) if use_axial else nn.Identity()
        self.mfds = MFDSBlock3D(out_ch)
        self.refine = BasicBlock3D(out_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.ms(x)
        x = self.axial(x)
        x = self.mfds(x)
        x = self.refine(x)
        return x


class FallbackSSM3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.gate = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        g = torch.sigmoid(self.gate(x))
        return y * g


class LMambaBlock3D(nn.Module):
    def __init__(self, channels: int, compress_ratio: int = 2):  # ★ 4 → 2: less bottleneck compression
        super().__init__()

        hidden = max(channels // compress_ratio, 16)

        self.in_proj = ConvNormAct3D(channels, hidden, 1, 1, 0)
        self.norm = nn.LayerNorm(hidden)

        try:
            from mamba_ssm import Mamba
            self.ssm = Mamba(d_model=hidden, d_state=16, d_conv=4, expand=2)
        except Exception:
            self.ssm = FallbackSSM3D(hidden)

        self.out_proj = nn.Conv3d(hidden, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.1))  # ★ 1e-3 → 0.1: LMamba contributes from epoch 1

    def forward(self, x):
        b, _, d, h, w = x.shape

        y = self.in_proj(x)
        ch = y.shape[1]

        seq = y.flatten(2).transpose(1, 2)
        seq = self.norm(seq)
        seq = self.ssm(seq)

        y = seq.transpose(1, 2).reshape(b, ch, d, h, w)
        y = self.out_proj(y)

        return x + self.gamma.to(dtype=x.dtype) * y


class CSDFBlock3D(nn.Module):
    def __init__(self, enc_ch: int, dec_ch: int, out_ch: int):
        super().__init__()

        self.enc_proj = ConvNormAct3D(enc_ch, out_ch, 1, 1, 0)
        self.dec_proj = ConvNormAct3D(dec_ch, out_ch, 1, 1, 0)

        # Gate with positive bias init → sigmoid(+2) ≈ 0.88 → initially preserves encoder skip
        gate_last = nn.Conv3d(out_ch, out_ch, 1)
        nn.init.constant_(gate_last.bias, 2.0)
        self.gate = nn.Sequential(
            nn.Conv3d(out_ch * 2, out_ch, 1),
            SafeGroupNorm(out_ch),
            nn.SiLU(inplace=True),
            gate_last,
            nn.Sigmoid(),
        )

        self.refine = BasicBlock3D(out_ch, out_ch)

    def forward(self, enc, dec):
        dec = match_spatial_shape(dec, enc)

        e = self.enc_proj(enc)
        d = self.dec_proj(dec)

        g = self.gate(torch.cat([e, d], dim=1))
        y = g * e + (1.0 - g) * d

        return self.refine(y)


class UpCSDFBlock3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, stride: Sequence[int]):
        super().__init__()

        stride = _to_3tuple(stride)

        self.up = nn.ConvTranspose3d(
            in_ch,
            out_ch,
            kernel_size=stride,
            stride=stride,
            bias=False,
        )

        self.csdf = CSDFBlock3D(skip_ch, out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = match_spatial_shape(x, skip)
        return self.csdf(skip, x)


class LMambaMFDSUNet3D(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        base_channels: int = 24,
        pool_strides: Union[List[Sequence[int]], Tuple[Sequence[int], ...]] = None,
        deep_supervision: bool = True,
        max_channels: int = 320,
    ):
        super().__init__()

        self.deep_supervision = bool(deep_supervision)

        if pool_strides is None or len(pool_strides) == 0:
            pool_strides = [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]

        pool_strides = [_to_3tuple(s) for s in pool_strides]
        pool_strides = [s for s in pool_strides if s != (1, 1, 1)]

        if len(pool_strides) < 2:
            raise ValueError(f"LMambaMFDSUNet3D requires at least 2 downsampling strides, got {pool_strides}")

        self.pool_strides = pool_strides
        num_down = len(pool_strides)

        level_channels = [min(base_channels * (2 ** i), max_channels) for i in range(num_down + 1)]

        self.stem = nn.Sequential(
            ConvNormAct3D(input_channels, level_channels[0], 3, 1, 1),
            BasicBlock3D(level_channels[0], level_channels[0]),
        )

        self.encoders = nn.ModuleList()
        for i, stride in enumerate(pool_strides):
            in_ch = level_channels[i]
            out_ch = level_channels[i + 1]

            self.encoders.append(
                MSFAEncoderBlock3D(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    stride=stride,
                    use_axial=(i >= 1),
                )
            )

        bottleneck_ch = level_channels[-1]
        self.bottleneck = nn.Sequential(
            LMambaBlock3D(bottleneck_ch),
            BasicBlock3D(bottleneck_ch, bottleneck_ch),
        )

        skip_channels = level_channels[:-1]
        decoder_strides = list(reversed(pool_strides))
        decoder_skip_channels = list(reversed(skip_channels))

        self.decoders = nn.ModuleList()
        self.ds_heads = nn.ModuleList()

        dec_in = bottleneck_ch
        for stride, skip_ch in zip(decoder_strides, decoder_skip_channels):
            out_ch = skip_ch
            self.decoders.append(
                UpCSDFBlock3D(
                    in_ch=dec_in,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                    stride=stride,
                )
            )
            self.ds_heads.append(nn.Conv3d(out_ch, num_classes, 1))
            dec_in = out_ch

        self.initialize_weights()

    def initialize_weights(self):
        """nnU-Net-style initialisation so only the network architecture differs
        from the base trainer.
        Uses 'leaky_relu' nonlinearity for kaiming init — SiLU has gain ~√2,
        similar to LeakyReLU."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def set_deep_supervision(self, enabled: bool):
        self.deep_supervision = bool(enabled)

    def forward(self, x):
        # ── encoder ──
        # Gradient checkpointing is applied to each encoder block and the
        # bottleneck.  This reduces peak activation memory by ~50-70% at the
        # cost of recomputing intermediate activations during backward.
        # It does NOT change the model's forward output, parameter count, or
        # optimizer trajectory — the fairness vs base nnUNet is preserved.
        skips = []

        x = self.stem(x)
        skips.append(x)

        for enc in self.encoders[:-1]:
            x = torch_checkpoint(enc, x, use_reentrant=False)
            skips.append(x)

        x = torch_checkpoint(self.encoders[-1], x, use_reentrant=False)
        x = torch_checkpoint(self.bottleneck, x, use_reentrant=False)

        # ── decoder ──
        # decoder_features are built in order: lowest-res first → highest-res last
        # Checkpoint decoder blocks too — UpCSDFBlock3D at C=256 still heavy
        decoder_features = []

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = torch_checkpoint(dec, x, skip, use_reentrant=False)
            decoder_features.append(x)

        # ds_heads were created in the same order as decoders:
        #   ds_heads[0]  ↔  decoder[0]  (lowest resolution)
        #   ds_heads[-1] ↔  decoder[-1] (highest resolution = full-res)

        # ── outputs: high → low resolution (nnU-Net convention) ──
        out_full_res = self.ds_heads[-1](decoder_features[-1])

        if not self.deep_supervision:
            return out_full_res

        # Build deep supervision list: [full_res, half_res, quarter_res, ...]
        # decoder_features[-2] is the second-highest resolution, etc.
        ds_outputs = [out_full_res]
        for feat, head in zip(
            reversed(decoder_features[:-1]),
            reversed(self.ds_heads[:-1]),
        ):
            ds_outputs.append(head(feat))

        return ds_outputs