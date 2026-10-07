"""
nnU-Net v2 Custom Model Base Chain Utilities
=============================================
Unified adaptation layer for all custom Trainer / Network implementations.
Ensures every custom model runs on the same nnU-Net plans-driven chain as
the base nnUNetTrainer: plans → features_per_stage → strides → kernel_sizes
→ norm (InstanceNorm3d) → nonlin (LeakyReLU) → init (He) → deep supervision
→ loss → optimizer → augmentation → validation → sliding window inference.

Author: SCI Experiment Framework
Date:   2026-05-18
"""

import os
import math
from typing import Union, Tuple, List, Dict, Any, Sequence

import torch
import torch.nn as nn

from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

# ---------------------------------------------------------------------------
# 1. Unified Plans Parameter Extractor
# ---------------------------------------------------------------------------

def get_nnunet_plan_params(
    plans_manager,          # PlansManager instance
    dataset_json: dict,
    configuration_manager,  # ConfigurationManager instance
    num_input_channels: int,
) -> Dict[str, Any]:
    """
    Extract all network-relevant parameters from nnU-Net plans in a unified
    way, so custom models can build themselves with the same hyperparameters
    as the base nnUNet.

    Returns a dict with keys:
      num_input_channels, num_classes, patch_size,
      pool_op_kernel_sizes, conv_kernel_sizes, features_per_stage,
      n_conv_per_stage_encoder, n_conv_per_stage_decoder, strides,
      deep_supervision_scales, dim, conv_op, norm_op, norm_op_kwargs,
      nonlin, nonlin_kwargs, UNet_base_num_features, unet_max_num_features
    """
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_classes = label_manager.num_segmentation_heads

    patch_size = configuration_manager.patch_size
    dim = len(patch_size)

    conv_op = convert_dim_to_conv_op(dim)
    instnorm = get_matching_instancenorm(dimension=dim)

    # --- Safe ConfigurationManager access (dict fallback for older nnU-Net) ---
    _conf = getattr(configuration_manager, 'configuration', {})

    def _cm(key, default=None):
        return getattr(configuration_manager, key, _conf.get(key, default))

    def _cm_list(key, default):
        v = _cm(key, None)
        return v if (v is not None and len(v) > 0) else default

    pool_op_kernel_sizes = _cm_list('pool_op_kernel_sizes', [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]])
    conv_kernel_sizes = _cm_list('conv_kernel_sizes', [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]])

    n_stages = len(conv_kernel_sizes)

    base_features = _cm('UNet_base_num_features', 32)
    max_features = _cm('unet_max_num_features', 320)
    features_per_stage = [
        min(base_features * (2 ** i), max_features)
        for i in range(n_stages)
    ]

    n_conv_per_stage_encoder = list(_cm_list('n_conv_per_stage_encoder', [2]*n_stages))
    n_conv_per_stage_decoder = list(_cm_list('n_conv_per_stage_decoder', [2]*(n_stages-1)))

    # --- Deep supervision scales ---
    import numpy as np
    if pool_op_kernel_sizes:
        cumprod_strides = np.cumprod(
            np.vstack(pool_op_kernel_sizes), axis=0
        )  # shape: (n_stages, dim)
        deep_supervision_scales = [
            list(1.0 / s) for s in cumprod_strides
        ][:-1]  # remove bottleneck
    else:
        deep_supervision_scales = []

    # --- Normalisation / Non-linearity (nnU-Net defaults) ---
    norm_op_name = instnorm.__module__ + "." + instnorm.__name__
    norm_op_kwargs = {"eps": 1e-5, "affine": True}

    nonlin_name = "torch.nn.LeakyReLU"
    nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}

    return {
        "num_input_channels": num_input_channels,
        "num_classes": num_classes,
        "patch_size": patch_size,
        "dim": dim,
        "pool_op_kernel_sizes": pool_op_kernel_sizes,
        "conv_kernel_sizes": conv_kernel_sizes,
        "features_per_stage": features_per_stage,
        "n_conv_per_stage_encoder": n_conv_per_stage_encoder,
        "n_conv_per_stage_decoder": n_conv_per_stage_decoder,
        "strides": pool_op_kernel_sizes,  # alias
        "deep_supervision_scales": deep_supervision_scales,
        "conv_op": conv_op,
        "norm_op": norm_op_name,
        "norm_op_kwargs": norm_op_kwargs,
        "nonlin": nonlin_name,
        "nonlin_kwargs": nonlin_kwargs,
        "UNet_base_num_features": base_features,
        "unet_max_num_features": max_features,
    }


# ---------------------------------------------------------------------------
# 2. BatchNorm3d Sanctuary ( NO BatchNorm3d in 3D small-batch training )
# ---------------------------------------------------------------------------

def sanitize_3d_norm(module: nn.Module) -> int:
    """
    Recursively replace every BatchNorm3d with InstanceNorm3d.
    Returns the number of replacements made.
    """
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            num_features = child.num_features
            new_norm = nn.InstanceNorm3d(num_features, affine=True)
            setattr(module, name, new_norm)
            replaced += 1
        else:
            replaced += sanitize_3d_norm(child)
    return replaced


def assert_no_batchnorm3d(module: nn.Module, model_name: str = "unknown") -> None:
    """Raise RuntimeError if any BatchNorm3d layer is found."""
    offenders = []
    for n, m in module.named_modules():
        if isinstance(m, nn.BatchNorm3d):
            offenders.append(n)
    if offenders:
        raise RuntimeError(
            f"[{model_name}] Found BatchNorm3d layers (will be unstable "
            f"under small-batch 3D training): {offenders}. "
            f"Call sanitize_3d_norm() before assert_no_batchnorm3d()."
        )


# ---------------------------------------------------------------------------
# 3. Parameter Counter
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# 4. Model Chain Summary Logger
# ---------------------------------------------------------------------------

def log_model_chain_summary(
    trainer,           # nnUNetTrainer instance (for logging)
    model: nn.Module,
    model_name: str,
    extra: Dict[str, Any] = None,
) -> None:
    """Print and save a structured summary of the model's nnU-Net chain."""
    params = count_parameters(model)
    has_bn = any(isinstance(m, nn.BatchNorm3d) for m in model.modules())
    has_wrapper = hasattr(model, "size_divisibility") or hasattr(model, "img_size")

    cm = trainer.configuration_manager
    pm = trainer.plans_manager

    summary_lines = [
        f"========== {model_name} Chain Summary ==========",
        f"  Dataset:      {pm.dataset_name}",
        f"  Config:       {trainer.configuration_name}",
        f"  Patch size:   {cm.patch_size}",
        f"  Batch size:   {cm.batch_size}",
        f"  Features:     {getattr(cm, 'configuration', {}).get('UNet_base_num_features', '?')} -> max {getattr(cm, 'configuration', {}).get('unet_max_num_features', '?')}",
        f"  Pool strides: {cm.pool_op_kernel_sizes}",
        f"  Conv kernels: {cm.conv_kernel_sizes}",
        f"  Deep sup.:    {trainer.enable_deep_supervision}",
        f"  Compile:      {trainer._do_i_compile()}",
        f"  Output fmt:   {'list (DS)' if trainer.enable_deep_supervision else 'single tensor'}",
        f"  Params:       {params['total']:,} total / {params['trainable']:,} trainable",
        f"  BatchNorm3d:  {'PRESENT (DANGER)' if has_bn else 'none (clean)'}",
        f"  ShapeWrapper: {'yes' if has_wrapper else 'none'}",
        f"  Init LR:      {trainer.initial_lr}",
        f"  Weight decay: {trainer.weight_decay}",
        f"  Optimizer:    {trainer.optimizer.__class__.__name__ if trainer.optimizer else 'not yet'}",
    ]
    if extra:
        for k, v in extra.items():
            summary_lines.append(f"  {k}:  {v}")
    summary_lines.append("=" * 50)

    for line in summary_lines:
        trainer.print_to_log_file(line)


# ---------------------------------------------------------------------------
# 5. Forward Debug Helper
# ---------------------------------------------------------------------------

def debug_forward_once(
    model: nn.Module,
    sample_input: torch.Tensor,
    model_name: str = "unknown",
    deep_supervision: bool = False,
) -> Dict[str, Any]:
    """
    Run a single forward pass and return diagnostic information.
    sample_input shape: [B, C, D, H, W]

    Returns dict with:
      ok, output_type, output_shape, ds_count, spatial_match,
      has_nan, has_inf, output_min, output_max, output_mean, output_std
    """
    result = {"ok": True, "model_name": model_name}

    with torch.no_grad():
        try:
            output = model(sample_input)
        except Exception as e:
            result["ok"] = False
            result["error"] = str(e)
            return result

    result["output_type"] = type(output).__name__

    if isinstance(output, (list, tuple)):
        result["ds_count"] = len(output)
        result["output_shape"] = str([o.shape for o in output])
        out0 = output[0]
    else:
        result["ds_count"] = 1
        result["output_shape"] = str(output.shape)
        out0 = output

    input_spatial = sample_input.shape[2:]
    output_spatial = tuple(out0.shape[2:])
    result["spatial_match"] = (input_spatial == output_spatial)
    result["spatial_input"] = input_spatial
    result["spatial_output"] = output_spatial

    result["has_nan"] = bool(torch.isnan(out0).any().item())
    result["has_inf"] = bool(torch.isinf(out0).any().item())
    result["output_min"] = float(out0.min().item())
    result["output_max"] = float(out0.max().item())
    result["output_mean"] = float(out0.mean().item())
    result["output_std"] = float(out0.std().item())

    return result


# ---------------------------------------------------------------------------
# 6. Lightweight Diagnostic Logging (env-var gated)
# ---------------------------------------------------------------------------

def _custom_debug_enabled() -> bool:
    return os.environ.get("NNUNET_CUSTOM_DEBUG", "").lower() in ("1", "true", "yes")


class CustomDebugContext:
    """
    Context manager that prints diagnostic info on the first few iterations
    when NNUNET_CUSTOM_DEBUG=1 is set.

    Usage in validation_step or train_step:
        with CustomDebugContext(model, batch, model_name, max_prints=3):
            pass  # actual forward happens inside the context
    """

    _print_counts: Dict[str, int] = {}

    def __init__(
        self,
        model: nn.Module,
        batch: dict,
        model_name: str,
        max_prints: int = 3,
        deep_supervision: bool = False,
    ):
        self.model = model
        self.batch = batch
        self.model_name = model_name
        self.max_prints = max_prints
        self.deep_supervision = deep_supervision
        self.enabled = _custom_debug_enabled()

    def __enter__(self):
        if not self.enabled:
            return self

        key = self.model_name
        cnt = self._print_counts.get(key, 0)
        self._do_print = cnt < self.max_prints
        if self._do_print:
            self._print_counts[key] = cnt + 1
        return self

    def __exit__(self, *args):
        pass

    def log(self, trainer_or_print, output, target, loss_val=None, grad_norm=None):
        """Call after forward to print diagnostics."""
        if not self.enabled or not self._do_print:
            return

        printer = (
            trainer_or_print.print_to_log_file
            if hasattr(trainer_or_print, "print_to_log_file")
            else print
        )

        printer(f"[DEBUG {self.model_name}] --- iteration #{self._print_counts.get(self.model_name, 0)} ---")

        # output
        if isinstance(output, (list, tuple)):
            printer(f"  DS outputs: {len(output)}")
            for i, o in enumerate(output):
                printer(f"    [{i}]: shape={o.shape}, min={o.min():.4f}, max={o.max():.4f}, "
                        f"mean={o.mean():.4f}, nan={torch.isnan(o).any().item()}, "
                        f"inf={torch.isinf(o).any().item()}")
            out0 = output[0]
        else:
            printer(f"  output: shape={output.shape}, min={output.min():.4f}, "
                    f"max={output.max():.4f}, nan={torch.isnan(output).any().item()}")
            out0 = output

        # prediction class distribution
        pred = out0.argmax(1)
        pred_unique, pred_counts = pred.unique(return_counts=True)
        pred_dict = dict(zip(pred_unique.tolist(), pred_counts.tolist()))
        printer(f"  pred classes: {pred_dict}")

        # foreground probability
        if out0.shape[1] > 1:
            probs = torch.softmax(out0, dim=1)
            for c in range(1, out0.shape[1]):
                fg_prob = probs[:, c]
                printer(f"  class {c} prob: min={fg_prob.min():.4f}, max={fg_prob.max():.4f}, "
                        f"mean={fg_prob.mean():.4f}")

        # target
        tgt = self.batch.get("target")
        if tgt is not None:
            if isinstance(tgt, list):
                tgt0 = tgt[0]
            else:
                tgt0 = tgt
            tgt_unique, tgt_counts = tgt0.unique(return_counts=True)
            tgt_dict = dict(zip(tgt_unique.tolist(), tgt_counts.tolist()))
            printer(f"  target classes: {tgt_dict}")

            if out0.shape[2:] != tgt0.shape[1:]:
                printer(f"  *** SHAPE MISMATCH: out_spatial={tuple(out0.shape[2:])} "
                        f"vs tgt_spatial={tuple(tgt0.shape[1:])}")

        # loss
        if loss_val is not None:
            printer(f"  loss: {loss_val}")

        # gradient norm
        if grad_norm is not None:
            printer(f"  grad_norm: {grad_norm:.4f}")

        # BatchNorm check
        bn_count = sum(1 for m in self.model.modules() if isinstance(m, nn.BatchNorm3d))
        if bn_count > 0:
            printer(f"  *** BatchNorm3d layers found: {bn_count}")

        # Mamba token count
        for name, module in self.model.named_modules():
            if "mamba" in name.lower() or "Mamba" in type(module).__name__:
                # try to get the last input shape from the module
                pass  # Mamba token log would need forward hook

        # gradient norm (if model is training)
        if self.model.training:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5 if total_norm > 0 else 0.0
            printer(f"  total grad norm: {total_norm:.4f}")

        printer(f"[DEBUG {self.model_name}] --- end ---")


# ---------------------------------------------------------------------------
# 7. Gate Initialisation Helpers (identity-friendly)
# ---------------------------------------------------------------------------

def init_gate_bias_to_positive(module: nn.Module, bias_value: float = 2.0):
    """
    Find the last Conv3d in every gate/sigmoid path and set its bias to a
    positive value so the initial gate output ≈ sigmoid(+2) ≈ 0.88.
    This biases the gate toward preserving the encoder skip.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv3d) and m.out_channels == 1:
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_value)


# ---------------------------------------------------------------------------
# 8. Deep Supervision Propagation Helper
# ---------------------------------------------------------------------------

def propagate_deep_supervision_toggle(network, enabled: bool):
    """
    Unwrap DDP → torch.compile → ShapeStrictWrapper → BaseChainFeatureAdapter3D
    and toggle the `deep_supervision` flag on the adapter.

    Also tries `model.deep_supervision` and `decoder.deep_supervision` as
    fallback targets (for nnU-Net base / UMamba style networks).
    """
    net = network
    if hasattr(net, 'module'):
        net = net.module
    if hasattr(net, '_orig_mod'):
        net = net._orig_mod

    # Walk through wrapper chain to find the adapter
    walked = [net]
    while hasattr(net, 'model'):
        net = net.model
        walked.append(net)
        if len(walked) > 20:
            break

    # Guard: for adapters with has_internal_ds=False (DS handled externally via
    # fallback logits pyramid), toggle the adapter directly and stop.  This
    # prevents accidentally enabling internal DS on models intentionally built
    # with deep_supervision=False (e.g., DynUNet with fallback_ds=True).
    # Walking past the adapter would reach the inner model's deep_supervision
    # attribute, which causes crashes with torch.compile due to DynUNet's
    # mutable-list side-effect data flow in its native DS path.
    for m in reversed(walked):
        if hasattr(m, 'has_internal_ds') and not m.has_internal_ds:
            m.deep_supervision = bool(enabled)
            return

    # Try BaseChainFeatureAdapter3D class-level deep_supervision
    for m in reversed(walked):
        if hasattr(m, 'deep_supervision') and 'deep_supervision' in type(m).__dict__:
            m.deep_supervision = bool(enabled)
            return

    # Fallback: try model.deep_supervision (e.g., LMambaMFDSNet with
    # has_internal_ds=True, or nnU-Net base / UMamba style networks)
    #
    # IMPORTANT: do NOT return after the first match.  For adapter-based
    # models (has_internal_ds=True) there may be TWO layers that need the
    # toggle — the BaseChainFeatureAdapter3D wrapper AND the inner model.
    # If we return early, the adapter's deep_supervision stays True and
    # the model returns a list-of-tensors during inference, which breaks
    # test-time mirror augmentation (torch.flip on a list → crash/hang).
    found_any = False
    for m in reversed(walked):
        if hasattr(m, 'deep_supervision') and not hasattr(m, 'size_divisibility'):
            if m.deep_supervision != bool(enabled):
                m.deep_supervision = bool(enabled)
            found_any = True
    if found_any:
        return

    # Last resort: set_deep_supervision method
    for m in reversed(walked):
        if hasattr(m, 'set_deep_supervision'):
            m.set_deep_supervision(enabled)
            return

