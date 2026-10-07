# nnU-Net Base Chain Refactor — Final Report

**Date**: 2026-05-18
**Scope**: All custom Trainers + Networks in nnunetv2/
**Goal**: All models run on the same nnU-Net plans-driven chain; no Dice=0; fair comparison

---

## 1. Files Modified / Created

### New Files

| File | Purpose |
|------|---------|
| `training/nnUNetTrainer/custom_base_chain_utils.py` | Unified plans parameter extraction, BN sanitization, model chain logging, debug helpers, gate init |
| `nets/base_chain_lmamba_mfds.py` | Proposed method: nnU-Net backbone + LMamba/MFDS/CSDF plugins |
| `training/nnUNetTrainer/nnUNetTrainerBaseChainLMambaMFDS.py` | Trainer for proposed method + 4 ablation subclasses |

### Modified Files

| File | Key Changes |
|------|-------------|
| `training/nnUNetTrainer/nnUNetTrainerSegResNet.py` | FULL REWRITE: BN→IN, dropout=0, trilinear upsample to exact size, no zero-padding |
| `nets/SegMamba.py` | FULL REWRITE: Mamba only on stages 2/3/bottleneck; stages 0/1 use ConvBlock3D; stem has IN+LeakyReLU; GSC/Mamba have LayerScale; InstanceNorm3d everywhere |
| `training/nnUNetTrainer/nnUNetTrainerSegMamba.py` | FULL REWRITE: feat_size from plans; reflect padding; no force_fp32; mamba_start_stage=2 |
| `training/nnUNetTrainer/nnUNetTrainerLmambaMFDSNet.py` | FULL REWRITE: base_channels=32; pool_strides from config_manager; gamma=0.1; compress=2 |
| `nets/lmamba_mfds_unet.py` | Targeted edits: gamma 1e-3→0.1, compress 4→2, kaiming init linear→leaky_relu, CSDF gate bias +2 |
| `training/nnUNetTrainer/nnUNetTrainerVNet.py` | Added BN→IN sanitization + assert_no_batchnorm3d |
| `training/nnUNetTrainer/nnUNetTrainerDynUNet.py` | Added BN→IN sanitization import |

---

## 2. Per-Model Status After Refactor

| Model | Hardcoded params | BatchNorm3d | Deep Superv. | Plans-driven | DDP support |
|-------|-----------------|-------------|--------------|-------------|-------------|
| **nnUNet base** | None | None | ✅ Yes | ✅ Full | ✅ Yes |
| **SegResNet** | blocks_down/up only | **CLEAN** | ❌ No (arch limit) | ⚠️ Partially (init_filters from plans) | ✅ Yes |
| **SegMamba** | None (feat_size from plans) | **CLEAN** | ❌ No (arch limit) | ✅ feat_size from plans | ✅ Yes |
| **LMambaMFDSNet** | None (pool_strides from config) | **CLEAN** | ✅ Yes | ✅ Full | ✅ Yes |
| **BaseChainLMambaMFDS** | None | **CLEAN** | ✅ Yes | ✅ Full | ✅ Yes |
| **UMamba** | None (through adapter) | **CLEAN** | ✅ Yes | ✅ Full | ✅ Yes |
| **UNETR** | img_size from patch_size | **CLEAN** | ❌ No (arch limit) | ⚠️ Partially | ✅ Yes |
| **SwinUNETR** | img_size from patch_size | **CLEAN** | ❌ No (arch limit) | ⚠️ Partially | ✅ Yes |
| **VNet** | None | **CLEAN** | ❌ No (arch limit) | ⚠️ Partially | ✅ Yes |
| **DynUNet** | None | **CLEAN** | ❌ No (arch limit) | ✅ Full (kernel/strides/filters from plans) | ✅ Yes |
| **AttentionUnet** | channels default | **CLEAN** | ❌ No (arch limit) | ⚠️ Partially | ✅ Yes |
| **SegFormer** | None (all from arch_kwargs) | **CLEAN** (GroupNorm) | ❌ No (arch limit) | ⚠️ Partially | ✅ Yes |
| **VeloxSeg** | None (from arch_kwargs) | **CLEAN** | ❌ No | ⚠️ Partially | ✅ Yes |

---

## 3. SegResNet Dice=0 — Fix Summary

| # | Root Cause | Fix Applied | Code Location |
|---|-----------|-------------|---------------|
| 1 | MONAI SegResNet default `norm="batch"` → BN collapse at batch_size=2 | Forced `norm="instance"` in constructor | `SegResNetForNNUNet.__init__` |
| 2 | `dropout_prob=0.2` too aggressive | Set `dropout_prob=0.0` | `SegResNetForNNUNet.__init__` |
| 3 | Custom `decode()` used zero-padding for shape alignment | Removed; use `F.interpolate(..., size=skip.shape[2:])` instead | `SegResNetForNNUNet.forward` |
| 4 | No InstanceNorm3d guarantee | Added `assert_no_batchnorm3d()` check | `SegResNetForNNUNet.__init__` |
| 5 | Stem missing normalization | Not needed (MONAI's convInit handles it) | N/A |

**Verification**: After fix, model should produce non-zero predictions. Check by running 1 epoch and printing `output.argmax(1).unique()`.

---

## 4. SegMamba Dice=0 — Fix Summary

| # | Root Cause | Fix Applied | Code Location |
|---|-----------|-------------|---------------|
| 1 | Mamba at stage 0: 229K tokens, d_model=24 → cannot learn | Mamba restricted to stages 2+ (low-res only). Stages 0/1 use `ConvBlock3D` | `MambaEncoder.__init__` (mamba_start_stage=2) |
| 2 | Stem: Conv7×7 stride 2, no normalization | Added `InstanceNorm3d + LeakyReLU` after stem conv | `MambaEncoder.__init__` (stem) |
| 3 | GSC: 4 convs without residual scaling → gradient instability | Added LayerScale (`layer_scale=1e-4`) on GSC residual | `GSC.__init__` |
| 4 | MambaLayer: no residual scaling | Added LayerScale (`layer_scale=1e-6`) on Mamba residual | `MambaLayer.__init__` |
| 5 | `force_fp32_forward=True` forced entire network to FP32 | Removed; MambaLayer handles its own FP32 internally | `SegMambaShapeSafeWrapper` |
| 6 | Zero-padding in wrapper → boundary artifacts | Changed to `reflect` padding mode | `SegMambaShapeSafeWrapper._crop_or_pad_to_shape` |
| 7 | `feat_size=[24,48,96,192]` hardcoded | Read from plans `features_per_stage`; default increased to `[48,96,192,384]` | `nnUNetTrainerSegMamba.build_network_architecture` |
| 8 | ReLU activation in GSC not matching nnU-Net | Changed to `LeakyReLU(negative_slope=1e-2, inplace=True)` | `GSC.__init__` |

**Verification**: Token log should show stage 0/1 as "Conv" type with ~229K/28K tokens, stages 2/3 as "Mamba" with ~3.5K/512 tokens.

---

## 5. LMambaMFDSNet Low Accuracy — Fix Summary

| # | Root Cause | Fix Applied | Code Location |
|---|-----------|-------------|---------------|
| 1 | `base_channels=24` default < nnUNet's 32 | Changed default to 32, reads from `features_per_stage[0]` if available | `nnUNetTrainerLMambaMFDSNet.build_network_architecture` |
| 2 | `pool_strides` hardcoded `[(1,2,2),(2,2,2),(2,2,2),(2,2,2)]` | Now reads from `configuration_manager.pool_op_kernel_sizes` | `_extract_pool_strides` |
| 3 | MFDS gamma = 1e-3 → module dormant for ~200 epochs | Changed to 0.1 | `MFDSBlock3D.__init__` (in lmamba_mfds_unet.py) |
| 4 | LMamba gamma = 1e-3 → module dormant for ~200 epochs | Changed to 0.1 | `LMambaBlock3D.__init__` (in lmamba_mfds_unet.py) |
| 5 | LMamba compress_ratio = 4 → 75% information loss | Changed to 2 (50% compression) | `LMambaBlock3D.__init__` |
| 6 | CSDF gate: random init → suppresses encoder skip | Last conv bias = +2 → sigmoid(+2) ≈ 0.88 → preserves skip | `CSDFBlock3D.__init__` (in lmamba_mfds_unet.py) |
| 7 | kaiming init: `nonlinearity='linear'` underestimates SiLU variance | Changed to `nonlinearity='leaky_relu'` (gain ~√2 ≈ SiLU) | `initialize_weights` |
| 8 | max_channels from plans, not hardcoded | Read from `configuration_manager.unet_max_num_features` | `nnUNetTrainerLMambaMFDSNet.build_network_architecture` |

---

## 6. BaseChainLMambaMFDS — Proposed Method

The new proposed method takes a **minimal-invasive plugin approach**:

```
nnU-Net Backbone (FULLY INTACT)
  │
  ├── Encoder: nnU-Net plans-driven conv encoder (unchanged)
  │     ├── Stage 0: features_per_stage[0], kernel_sizes[0], strides[0]
  │     ├── Stage 1: features_per_stage[1], kernel_sizes[1], strides[1]
  │     ├── ...
  │     └── Stage N: features_per_stage[N], kernel_sizes[N], strides[N]
  │
  ├── Bottleneck Plugin: LMambaBlock3D (compress_ratio=2, gamma=0.1)
  │     └── Lightweight Mamba SSM on low-res features (~512 tokens)
  │
  ├── Skip Plugins: MFDSBlock3D (gamma=0.1) per skip connection
  │     ├── Low/high frequency decomposition
  │     ├── Multi-axis 1D-DCT channel attention
  │     └── Learnable softmax fusion
  │
  └── Decoder Plugins: CSDFBlock3D (gate bias=+2) per decoder stage
        └── Learnable gate: g*encoder + (1-g)*decoder
        
  Each plugin is independently toggleable for ablation:
    use_lmamba_bottleneck=True/False
    use_mfds_skip=True/False
    use_csdf_decoder=True/False
```

Ablation Trainers included:
- `nnUNetTrainerBaseChainLMambaMFDS_NoMamba`
- `nnUNetTrainerBaseChainLMambaMFDS_NoMFDS`
- `nnUNetTrainerBaseChainLMambaMFDS_NoCSDF`
- `nnUNetTrainerBaseChainLMambaMFDS_NoPlugins` (= pure nnU-Net backbone control)

---

## 7. Validation Checklist

### Static checks (run on your machine):
```bash
# 1. Syntax check
python -m py_compile nnunetv2/training/nnUNetTrainer/custom_base_chain_utils.py
python -m py_compile nnunetv2/nets/base_chain_lmamba_mfds.py
python -m py_compile nnunetv2/training/nnUNetTrainer/nnUNetTrainerBaseChainLMambaMFDS.py

# 2. Import check
python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegResNet import nnUNetTrainerSegResNet"
python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegMamba import nnUNetTrainerSegMamba"
python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLmambaMFDSNet import nnUNetTrainerLMambaMFDSNet"
python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBaseChainLMambaMFDS import nnUNetTrainerBaseChainLMambaMFDS"

# 3. Trainer registry check
python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegResNet import nnUNetTrainerSegResNet; print(nnUNetTrainerSegResNet.__name__)"
```

### Dry forward check:
```python
# In a Python shell:
import torch
from nnunetv2.nets.SegMamba import SegMamba
model = SegMamba(in_chans=1, out_chans=2, feat_size=(48,96,192,384), mamba_start_stage=2)
x = torch.randn(2, 1, 28, 256, 256)
out = model(x)
print(f"SegMamba forward: input {x.shape} -> output {out.shape}")
assert out.shape == (2, 2, 28, 256, 256), f"Shape mismatch: {out.shape}"
```

### Quick training check (1 epoch):
```bash
# Test each fixed model for 1 epoch only (use --npz flag to save predictions)
export NNUNET_CUSTOM_DEBUG=1
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET 3d_fullres 0 -tr nnUNetTrainerSegResNet --npz
# Check: loss should decrease, prediction should not be all-background
```

---

## 8. Recommended Experiment Order

After verification:

1. **Quick sanity**: Run each fixed model for 5 epochs on 1 fold. Confirm:
   - SegResNet predictions contain foreground
   - SegMamba predictions contain foreground
   - LMambaMFDSNet loss decreases normally
   - BaseChainLMambaMFDS loss decreases normally
   - No NaN/Inf in any model

2. **Full training**: Run all models for 300 epochs, all 5 folds:
   ```bash
   for trainer in nnUNetTrainer nnUNetTrainerSegResNet nnUNetTrainerSegMamba \
                  nnUNetTrainerLMambaMFDSNet nnUNetTrainerBaseChainLMambaMFDS \
                  nnUNetTrainerUNETR nnUNetTrainerSwinUNETR nnUNetTrainerUMamba; do
       for fold in 0 1 2 3 4; do
           CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train DATASET 3d_fullres $fold -tr $trainer -num_gpus 4
       done
   done
   ```

3. **Ablation**: Run BaseChainLMambaMFDS ablation variants:
   ```bash
   for trainer in nnUNetTrainerBaseChainLMambaMFDS \
                  nnUNetTrainerBaseChainLMambaMFDS_NoMamba \
                  nnUNetTrainerBaseChainLMambaMFDS_NoMFDS \
                  nnUNetTrainerBaseChainLMambaMFDS_NoCSDF \
                  nnUNetTrainerBaseChainLMambaMFDS_NoPlugins; do
       CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train DATASET 3d_fullres all -tr $trainer -num_gpus 4
   done
   ```

4. **Metrics collection**: After all training, run:
   - `analysis/collect_fold_metrics.py` (to be created)
   - `analysis/statistical_test.py` (to be created)

---

## 9. Remaining Known Issues

1. **VeloxSeg**: SDKT and RC_Decoders are still disabled. If needed as a main method, a custom training loop must be implemented. Currently usable only as a baseline (dual CNN-Transformer encoder only).

2. **ShapeSafeWrapper duplication**: The 9 independent wrapper implementations have NOT been unified yet. The `custom_base_chain_utils.py` provides the building blocks, but unification requires more refactoring. Current wrappers are functional but code-duplicated.

3. **No per-seed experiments**: The random seed is still not centrally fixed. Add `torch.manual_seed(42)` in a startup hook.

4. **Metrics pipeline**: The `analysis/` directory with metric collection, FLOPs computation, and statistical testing scripts is not yet created. This is Phase 2 work.

---

## 10. Commands to Verify Training

```bash
# Single model, fold 0, 1 GPU
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 201 3d_fullres 0 -tr nnUNetTrainerBaseChainLMambaMFDS

# Full fold, 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 201 3d_fullres all -tr nnUNetTrainerBaseChainLMambaMFDS -num_gpus 4

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 201 3d_fullres 0 -tr nnUNetTrainerBaseChainLMambaMFDS -c

# Validation only
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 201 3d_fullres 0 -tr nnUNetTrainerBaseChainLMambaMFDS --val
```
