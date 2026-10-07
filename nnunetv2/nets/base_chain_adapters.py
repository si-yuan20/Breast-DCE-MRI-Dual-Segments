"""
Base Chain Adapters — Unified nnU-Net v2 Model Adaptation Layer
================================================================
Makes ANY 3D segmentation model output a nnU-Net-compatible Deep
Supervision list, driven entirely by nnU-Net plans.

Modules:
  PlansDrivenConfig          — unified plans parameter extractor
  NormSanitizer3D            — BatchNorm3d → InstanceNorm3d
  ShapeStrictWrapper3D       — reflect-pad wrapper (no zero-pad)
  DeepSupervisionHead3D      — DS head from features OR logits pyramid
  BaseChainFeatureAdapter3D  — wraps any model → DS list output

All models that go through this adapter will:
  - output list[Tensor] matching _get_deep_supervision_scales()
  - output[0] spatial == input spatial
  - have NO BatchNorm3d
  - read all params from plans (no hardcoded channels/strides)
"""

import math
import numpy as np
from typing import Union, List, Tuple, Dict, Any, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# 1. PlansDrivenConfig
# ===========================================================================

class PlansDrivenConfig:
    """
    Unified extraction of ALL nnU-Net plans parameters needed to build or
    adapt a network.

    Usage:
        cfg = PlansDrivenConfig.from_nnunet(
            plans_manager, dataset_json, configuration_manager, num_input_channels
        )
        print(cfg.features_per_stage)
    """

    def __init__(self):
        self.num_input_channels: int = 1
        self.num_classes: int = 2
        self.patch_size: Tuple[int, ...] = (64, 64, 64)
        self.dim: int = 3
        self.pool_op_kernel_sizes: List[Tuple[int, ...]] = []
        self.conv_kernel_sizes: List[Tuple[int, ...]] = []
        self.features_per_stage: List[int] = []
        self.n_conv_per_stage_encoder: List[int] = []
        self.n_conv_per_stage_decoder: List[int] = []
        self.deep_supervision_scales: List[List[float]] = []
        self.n_stages: int = 6
        self.UNet_base_num_features: int = 32
        self.unet_max_num_features: int = 320
        self.batch_size: int = 2
        self.dataset_name: str = "unknown"
        self.configuration_name: str = "3d_fullres"

    @classmethod
    def from_nnunet(
        cls,
        plans_manager,
        dataset_json: dict,
        configuration_manager,
        num_input_channels: int,
    ) -> "PlansDrivenConfig":
        cfg = cls()

        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

        label_manager = plans_manager.get_label_manager(dataset_json)

        cfg.num_input_channels = num_input_channels
        cfg.num_classes = label_manager.num_segmentation_heads
        cfg.patch_size = tuple(configuration_manager.patch_size)
        cfg.dim = len(cfg.patch_size)
        cfg.dataset_name = plans_manager.dataset_name
        cfg.configuration_name = getattr(configuration_manager, "configuration_name", "3d_fullres")

        # ── Safe ConfigurationManager access ──
        # Server's nnU-Net stores params in self.configuration dict, NOT as
        # direct attributes. We try the attribute first, then fall back to dict.
        _conf = getattr(configuration_manager, 'configuration', {})

        def _cm(key, default=None):
            return getattr(configuration_manager, key, _conf.get(key, default))

        def _cm_list(key, default):
            v = _cm(key, None)
            return v if (v is not None and len(v) > 0) else default

        # ── Pool / Conv topology ──
        pool = _cm_list('pool_op_kernel_sizes', [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]])
        cfg.pool_op_kernel_sizes = [tuple(int(x) for x in s) for s in pool]

        conv = _cm_list('conv_kernel_sizes', [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]])
        cfg.conv_kernel_sizes = [tuple(int(x) for x in s) for s in conv]

        cfg.n_stages = len(cfg.conv_kernel_sizes)

        # ── Features ──
        base = _cm('UNet_base_num_features', 32)
        maxf = _cm('unet_max_num_features', 320)
        cfg.UNet_base_num_features = base
        cfg.unet_max_num_features = maxf
        cfg.features_per_stage = [
            min(base * (2 ** i), maxf) for i in range(cfg.n_stages)
        ]

        # ── Convs per stage ──
        cfg.n_conv_per_stage_encoder = list(_cm_list('n_conv_per_stage_encoder', [2]*cfg.n_stages))
        cfg.n_conv_per_stage_decoder = list(_cm_list('n_conv_per_stage_decoder', [2]*(cfg.n_stages-1)))

        # ── Batch size ──
        cfg.batch_size = _cm('batch_size', 2)

        # ── Deep supervision scales ──
        if pool:
            cumprod = np.cumprod(np.vstack(pool), axis=0)  # (n_stages, dim)
            scales = (1.0 / cumprod).tolist()
            cfg.deep_supervision_scales = [list(s) for s in scales[:-1]]  # remove bottleneck
        else:
            cfg.deep_supervision_scales = []

        return cfg

    @property
    def num_ds_scales(self) -> int:
        return len(self.deep_supervision_scales)

    @property
    def required_size_divisibility(self) -> Tuple[int, ...]:
        """Minimum spatial divisibility for all encoder/decoder stages."""
        div = [1] * self.dim
        for stride in self.pool_op_kernel_sizes:
            for d in range(self.dim):
                div[d] = max(div[d], int(stride[d]))
        # Ensure at least 2x the stride product for safety
        prod = [1] * self.dim
        for stride in self.pool_op_kernel_sizes:
            for d in range(self.dim):
                prod[d] *= int(stride[d])
        return tuple(max(d, p) for d, p in zip(div, prod))

    def summary(self) -> str:
        lines = [
            f"PlansDrivenConfig(dataset={self.dataset_name}, config={self.configuration_name})",
            f"  in_ch={self.num_input_channels}  out_ch={self.num_classes}",
            f"  patch={self.patch_size}  batch={self.batch_size}",
            f"  stages={self.n_stages}  features={self.features_per_stage}",
            f"  pool={self.pool_op_kernel_sizes}",
            f"  conv_kernels={self.conv_kernel_sizes}",
            f"  DS scales={self.deep_supervision_scales}  (n={self.num_ds_scales})",
        ]
        return "\n".join(lines)


# ===========================================================================
# 2. NormSanitizer3D
# ===========================================================================

class NormSanitizer3D:
    """Recursively replace every BatchNorm3d → InstanceNorm3d(affine=True)."""

    @staticmethod
    def replace_bn(module: nn.Module, verbose: bool = True) -> int:
        replaced = 0
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm3d):
                num_features = child.num_features
                new_norm = nn.InstanceNorm3d(num_features, affine=True)
                setattr(module, name, new_norm)
                replaced += 1
            else:
                replaced += NormSanitizer3D.replace_bn(child, verbose=False)
        if verbose and replaced > 0:
            # Find first parent that has a meaningful class name
            cls_name = type(module).__name__
            if cls_name not in ("ModuleList", "Sequential", "Module"):
                print(f"[NormSanitizer] {cls_name}: replaced {replaced} BatchNorm3d → InstanceNorm3d")
        return replaced

    @staticmethod
    def assert_clean(module: nn.Module, model_name: str = "unknown") -> None:
        offenders = []
        for n, m in module.named_modules():
            if isinstance(m, nn.BatchNorm3d):
                offenders.append(n)
        if offenders:
            raise RuntimeError(
                f"[{model_name}] BatchNorm3d layers still present: {offenders}. "
                f"Call NormSanitizer3D.replace_bn() first."
            )


# ===========================================================================
# 3. ShapeStrictWrapper3D
# ===========================================================================

class ShapeStrictWrapper3D(nn.Module):
    """
    Minimal shape-safe wrapper.  Prefers NO padding; if required, uses
    REFLECT padding (never zero).  Logs every pad/crop operation.
    """

    def __init__(
        self,
        model: nn.Module,
        size_divisibility: Union[int, Tuple[int, ...]] = 1,
        minimum_spatial_size: Union[int, Tuple[int, ...]] = 1,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        self.model = model
        self.padding_mode = padding_mode

        if isinstance(size_divisibility, int):
            self.size_divisibility = (int(size_divisibility),) * 3
        else:
            self.size_divisibility = tuple(int(i) for i in size_divisibility)

        if isinstance(minimum_spatial_size, int):
            self.minimum_spatial_size = (int(minimum_spatial_size),) * 3
        else:
            self.minimum_spatial_size = tuple(int(i) for i in minimum_spatial_size)

        self._pad_log: List[str] = []

    @staticmethod
    def _compute_safe_shape(orig, divisibility, minimum) -> Tuple[int, ...]:
        out = []
        for o, d, m in zip(orig, divisibility, minimum):
            target = max(int(o), int(m))
            div = max(1, int(d))
            target = int(math.ceil(target / div) * div)
            out.append(target)
        return tuple(out)

    @staticmethod
    def _pad_tensor(x: torch.Tensor, orig: Tuple[int, ...], target: Tuple[int, ...], mode: str) -> torch.Tensor:
        if tuple(x.shape[2:]) == target:
            return x
        cur = tuple(x.shape[2:])
        # crop
        slices = [slice(None), slice(None)]
        for cs, ts in zip(cur, target):
            if cs > ts:
                st = (cs - ts) // 2
                slices.append(slice(st, st + ts))
            else:
                slices.append(slice(None))
        x = x[tuple(slices)]
        # pad
        pads = []
        for cs, ts in zip(reversed(tuple(x.shape[2:])), reversed(target)):
            if cs < ts:
                diff = ts - cs
                pads.extend([diff // 2, diff - diff // 2])
            else:
                pads.extend([0, 0])
        if any(pads):
            x = F.pad(x, pads, mode=mode)
        return x

    @staticmethod
    def _crop_only(x: torch.Tensor, target: Tuple[int, ...]) -> torch.Tensor:
        """Crop (never pad) for output restoration."""
        cur = tuple(x.shape[2:])
        if cur == target:
            return x
        slices = [slice(None), slice(None)]
        for cs, ts in zip(cur, target):
            if cs > ts:
                st = (cs - ts) // 2
                slices.append(slice(st, st + ts))
            else:
                slices.append(slice(None))
        return x[tuple(slices)]

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        orig = tuple(x.shape[2:])
        safe = self._compute_safe_shape(orig, self.size_divisibility, self.minimum_spatial_size)

        # Log padding if needed
        pad_amount = sum(s - o for s, o in zip(safe, orig) if s > o)
        if pad_amount > 0 and len(self._pad_log) < 5:
            self._pad_log.append(f"ShapeStrict: pad {pad_amount} voxels (orig={orig} safe={safe})")

        if orig != safe:
            x = self._pad_tensor(x, orig, safe, mode=self.padding_mode)

        out = self.model(x)

        # Restore spatial shape
        if isinstance(out, (list, tuple)):
            restored = []
            for o in out:
                restored.append(self._crop_only(o, orig))
            return restored
        else:
            return self._crop_only(out, orig)


# ===========================================================================
# 4. DeepSupervisionHead3D
# ===========================================================================

class DeepSupervisionHead3D(nn.Module):
    """
    Produces DS outputs for a model.

    Two modes:
      REAL (preferred):    Extract intermediate decoder features via a
                           user-provided `feature_getter` callable, then
                           apply 1×1×1 conv heads.

      FALLBACK (logits pyramid):
                           Take the full-resolution logits, downsample to
                           each DS scale, and apply 1×1×1 conv heads.
                           Marked as fallback_ds=True in logs.
    """

    def __init__(
        self,
        num_classes: int,
        ds_scales: List[List[float]],
        feature_channels: Optional[List[int]] = None,
        feature_getter: Optional[Callable[[], List[torch.Tensor]]] = None,
        fallback_ds: bool = False,
    ):
        """
        Args:
            num_classes:       number of output classes
            ds_scales:         list of [scale_D, scale_H, scale_W] for each DS level
            feature_channels:  channel counts for intermediate features (len = n_scales-1)
                               Only needed for REAL mode.
            feature_getter:    callable that returns list of intermediate feature tensors
                               [feat_1/2, feat_1/4, ...] (no full-res needed)
                               Only needed for REAL mode.
            fallback_ds:       if True, use logits pyramid fallback regardless of
                               feature_getter availability.
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.ds_scales = ds_scales
        self.n_scales = len(ds_scales)
        self.fallback_ds = bool(fallback_ds)

        # Full-resolution head (always needed)
        self.head_full = nn.Conv3d(self.num_classes, self.num_classes, 1, bias=True)

        if feature_getter is not None and feature_channels is not None and not fallback_ds:
            # ── REAL DS mode: use intermediate features ──
            self._mode = "real"
            self._feature_getter = feature_getter
            self._feature_channels = list(feature_channels)
            self.ds_heads = nn.ModuleList()
            for ch in self._feature_channels:
                self.ds_heads.append(nn.Conv3d(int(ch), self.num_classes, 1, bias=True))
        else:
            # ── FALLBACK DS mode: downsample logits ──
            self._mode = "fallback"
            self._feature_getter = None
            self.ds_heads = nn.ModuleList()
            for _ in range(self.n_scales - 1):
                self.ds_heads.append(nn.Conv3d(self.num_classes, self.num_classes, 1, bias=True))

        # Apply nnU-Net style initialization
        for head in self.ds_heads:
            nn.init.kaiming_normal_(head.weight, mode='fan_out', nonlinearity='leaky_relu', a=1e-2)
            if head.bias is not None:
                nn.init.constant_(head.bias, 0)

    @property
    def mode(self) -> str:
        return self._mode

    def forward(self, full_res_logits: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            full_res_logits: [B, C, D, H, W] — the full-resolution segmentation logits
                             from the model's forward pass.

        Returns:
            list of [B, C, D_s, H_s, W_s] at each DS scale, full-res first.
        """
        outputs = [self.head_full(full_res_logits)]

        if self._mode == "real" and self._feature_getter is not None:
            features = self._feature_getter()
            # features[k] → DS scale k+1
            for i, feat in enumerate(features):
                if i < len(self.ds_heads):
                    out = self.ds_heads[i](feat)
                    outputs.append(out)
        else:
            # Fallback: downsample full-res logits
            for i, scale in enumerate(self.ds_scales[1:], 1):
                down = F.interpolate(
                    full_res_logits,
                    scale_factor=scale,
                    mode="trilinear" if len(scale) == 3 else "bilinear",
                    align_corners=False,
                )
                out = self.ds_heads[i - 1](down)
                outputs.append(out)

        return outputs


# ===========================================================================
# 5. BaseChainFeatureAdapter3D
# ===========================================================================

class BaseChainFeatureAdapter3D(nn.Module):
    """
    Universal wrapper: transforms ANY model's output into a nnU-Net
    Deep Supervision list.

    Architecture:
      model(x) → full_res_logits [B, C, D, H, W]
        → DSHead3D  → [full, half, quarter, ...]
    """

    def __init__(
        self,
        model: nn.Module,
        ds_head: DeepSupervisionHead3D,
        has_internal_ds: bool = False,
    ):
        """
        Args:
            model:            the underlying segmentation model
            ds_head:          DeepSupervisionHead3D instance
            has_internal_ds:  if True, the model already outputs a DS list;
                              ds_head is NOT applied (pass-through).
        """
        super().__init__()
        self.model = model
        self.ds_head = ds_head
        self.has_internal_ds = bool(has_internal_ds)
        self.deep_supervision = True  # toggled by Trainer.set_deep_supervision_enabled

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.model(x)

        if self.has_internal_ds:
            # Model already outputs DS list [full, half, ...]
            if isinstance(out, (list, tuple)):
                ds_list = list(out)
            else:
                ds_list = self.ds_head(out)
        else:
            # Model outputs single tensor → apply DS head
            if isinstance(out, (list, tuple)):
                full_res = out[0]
            else:
                full_res = out
            ds_list = self.ds_head(full_res)

        # Inference / validation with DS off: return single full-res tensor
        if not self.deep_supervision:
            return ds_list[0]

        return ds_list


# ===========================================================================
# 6. Convenience: Build a fully adapted model from Plans
# ===========================================================================

def adapt_model_to_nnunet_base_chain(
    raw_model: nn.Module,
    cfg: PlansDrivenConfig,
    model_name: str = "unknown",
    feature_getter: Optional[Callable[[], List[torch.Tensor]]] = None,
    feature_channels: Optional[List[int]] = None,
    has_internal_ds: bool = False,
    fallback_ds: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Full adaptation pipeline: BN sanitize → shape wrapper → DS adapter.

    Returns a model that outputs a DS list compatible with nnU-Net.
    """
    # 1. Ban BatchNorm3d
    replaced = NormSanitizer3D.replace_bn(raw_model, verbose=verbose)
    if verbose and replaced > 0:
        print(f"[{model_name}] NormSanitizer: {replaced} BatchNorm3d → InstanceNorm3d")
    NormSanitizer3D.assert_clean(raw_model, model_name)

    # 2. Shape wrapper (reflect pad)
    wrapped = ShapeStrictWrapper3D(
        raw_model,
        size_divisibility=cfg.required_size_divisibility,
        minimum_spatial_size=cfg.required_size_divisibility,
        padding_mode="reflect",
    )

    # 3. DS head
    ds_head = DeepSupervisionHead3D(
        num_classes=cfg.num_classes,
        ds_scales=cfg.deep_supervision_scales,
        feature_channels=feature_channels,
        feature_getter=feature_getter,
        fallback_ds=fallback_ds or (feature_getter is None),
    )

    # 4. Adapter
    adapter = BaseChainFeatureAdapter3D(
        model=wrapped,
        ds_head=ds_head,
        has_internal_ds=has_internal_ds,
    )

    if verbose:
        ds_mode = ds_head.mode
        print(
            f"[{model_name}] Adapted: DS mode={ds_mode}, "
            f"scales={cfg.num_ds_scales}, "
            f"classes={cfg.num_classes}, "
            f"divisibility={cfg.required_size_divisibility}"
        )

    return adapter
