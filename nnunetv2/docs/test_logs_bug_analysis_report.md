# test_logs_20260518_144930 — Bug Analysis & Fix Report

**Date**: 2026-05-18
**Log Directory**: `test_logs_20260518_144930/`
**Tools Source**: `test_all_networks.sh` (diagnose mode)

---

## 0. Log Summary

| Trainer | Status | Root Cause |
|---------|--------|------------|
| nnUNetTrainer | ⚠️ AttributeError (non-fatal) | `UNet_base_num_features` not an attribute |
| nnUNetTrainerSegResNet | ❌ ImportError | **POISONED by SegFormer — see below** |
| nnUNetTrainerSegMamba | ❌ ImportError | **POISONED by SegFormer — see below** |
| nnUNetTrainerLMambaMFDSNet | ⚠️ AttributeError (non-fatal) | `UNet_base_num_features` not an attribute |
| nnUNetTrainerBaseChainLMambaMFDS | ❌ KeyError | `use_lmamba_bottleneck` in `my_init_kwargs` |
| nnUNetTrainerUNETR | ❌ ImportError | **POISONED by SegFormer — see below** |
| nnUNetTrainerSwinUNETR | ❌ ImportError | **POISONED by SegFormer — see below** |
| nnUNetTrainerVNet | ❌ ImportError | **POISONED by SegFormer — see below** |
| nnUNetTrainerDynUNet | ❌ ImportError | **POISONED by SegFormer — see below** |
| nnUNetTrainerAttentionUnet | ❌ ImportError | **POISONED by SegFormer — see below** |
| nnUNetTrainerSegFormer | ❌ ImportError | `RobustSegFormer3D` does not exist |
| nnUNetTrainerVeloxSeg | ❌ ImportError | **POISONED by SegFormer — see below** |
| nnUNetTrainerUMamba | ❌ ImportError | **POISONED by SegFormer — see below** |

**Successful runs**: **0 out of 13** trainers completed successfully.

---

## 1. Bug #1 (P0 — BLOCKER): SegFormer import poisons ALL trainer discovery

### Evidence
Every single trainer that failed with `ImportError` shows the SAME traceback:

```
File ".../nnunetv2/training/nnUNetTrainer/nnUNetTrainerSegFormer.py", line 5, in <module>
    from nnunetv2.nets.segformer3d import RobustSegFormer3D
ImportError: cannot import name 'RobustSegFormer3D' from 'nnunetv2.nets.segformer3d'
```

This happens even for **completely unrelated** trainers like `nnUNetTrainerUNETR`, `nnUNetTrainerSegMamba`, `nnUNetTrainerVNet`, `nnUNetTrainerUMamba`.

### Root Cause

`nnU-Net`'s `recursive_find_python_class()` uses `pkgutil.iter_modules()` to enumerate ALL `.py` files in the `nnUNetTrainer/` directory, then does `importlib.import_module()` on EACH one to find the requested trainer class. This means every Trainer module is imported — even ones the user didn't ask for.

When `nnUNetTrainerSegFormer.py` has a broken import at module level, ALL trainers become unreachable.

### How it poisons every trainer
```
User runs: nnUNetv2_train 305 3d_fullres 0 -tr nnUNetTrainerSegMamba

recursive_find_python_class iterates through ALL trainer modules:
  ├── nnUNetTrainer.py          → imports OK
  ├── nnUNetTrainerAttentionUnet.py → imports OK
  ├── nnUNetTrainerSegFormer.py → ❌ ImportError: RobustSegFormer3D not found
  └── nnUNetTrainerSegMamba.py  → NEVER REACHED
Result: Trainer discovery FAILS before finding SegMamba
```

### Fix Applied
Changed `nnUNetTrainerSegFormer.py` line 5:
```python
# OLD (broken):
from nnunetv2.nets.segformer3d import RobustSegFormer3D

# NEW (fixed):
from nnunetv2.nets.segformer3d import SegFormer3D
```

Also updated the class usage from `RobustSegFormer3D(...)` → `SegFormer3D(...)`.

---

## 2. Bug #2 (P1): `ConfigurationManager` has no attribute `UNet_base_num_features`

### Evidence
```
AttributeError: 'ConfigurationManager' object has no attribute 'UNet_base_num_features'
```
Found in logs: `diagnose_nnUNetTrainer.log`, `diagnose_nnUNetTrainerLMambaMFDSNet.log`

### Root Cause
The server's nnU-Net version stores `UNet_base_num_features` inside `ConfigurationManager.configuration` dict, not as a direct attribute. Our code accessed it as `configuration_manager.UNet_base_num_features` (direct attribute) which fails.

This affects:
1. `PlansDrivenConfig.from_nnunet()` in `base_chain_adapters.py` lines 94-95
2. `get_nnunet_plan_params()` in `custom_base_chain_utils.py` lines 60-61
3. `log_model_chain_summary()` in `custom_base_chain_utils.py` line 181

### Fix Applied
Changed all three locations from:
```python
# OLD (broken on older nnU-Net):
base = configuration_manager.UNet_base_num_features
maxf = configuration_manager.unet_max_num_features
```
To:
```python
# NEW (compatible with all nnU-Net versions):
conf = getattr(configuration_manager, 'configuration', {})
base = conf.get('UNet_base_num_features', 32)
maxf = conf.get('unet_max_num_features', 320)
```

---

## 3. Bug #3 (P0 — BLOCKER): `BaseChainLMambaMFDS` KeyError on extra kwargs

### Evidence
```
File ".../nnUNetTrainer.py", line 112, in __init__
    self.my_init_kwargs[k] = locals()[k]
KeyError: 'use_lmamba_bottleneck'
```

### Root Cause
The base `nnUNetTrainer.__init__` saves all init parameters for checkpointing:
```python
for k in inspect.signature(self.__init__).parameters.keys():
    self.my_init_kwargs[k] = locals()[k]
```

`inspect.signature(self.__init__)` inspects the subclass's (`nnUNetTrainerBaseChainLMambaMFDS`) `__init__` method because `self` is an instance of that subclass. The subclass signature included `use_lmamba_bottleneck`, `use_mfds_skip`, `use_csdf_decoder`, `lmamba_compress_ratio`. But `locals()` in the **base** `nnUNetTrainer.__init__` frame only contains `self, plans, configuration, fold, dataset_json, device` — not the subclass's extra kwargs. Hence `KeyError`.

### Fix Applied
Replaced explicit extra kwargs in `__init__` signature with `**kwargs`:
```python
# OLD (broken — extra kwargs leak into my_init_kwargs):
def __init__(self, plans, configuration, fold, dataset_json, device=device,
             use_lmamba_bottleneck=True, use_mfds_skip=True, ...):

# NEW (fixed — pops config before super().__init__):
def __init__(self, plans, configuration, fold, dataset_json, device=device,
             **kwargs):
    self._use_lmamba = kwargs.pop('use_lmamba_bottleneck', True)
    self._use_mfds = kwargs.pop('use_mfds_skip', True)
    self._use_csdf = kwargs.pop('use_csdf_decoder', True)
    self._compress = kwargs.pop('lmamba_compress_ratio', 2)
    super().__init__(plans, configuration, fold, dataset_json, device=device)
```

---

## 4. Files Modified in This Fix

| File | Bug | Change |
|------|-----|--------|
| `nnUNetTrainerSegFormer.py` | #1 | `RobustSegFormer3D` → `SegFormer3D` |
| `base_chain_adapters.py` L94-100 | #2 | `cfg.UNet_base_num_features` → `conf.get('UNet_base_num_features', 32)` |
| `custom_base_chain_utils.py` L60-63 | #2 | Same fix in `get_nnunet_plan_params` |
| `custom_base_chain_utils.py` L181 | #2 | Same fix in `log_model_chain_summary` |
| `nnUNetTrainerBaseChainLMambaMFDS.py` L17-24 | #3 | Extra kwargs → `**kwargs.pop(...)` before `super().__init__` |

---

## 5. Verification Commands (run on server after deploying fixes)

```bash
# 1. Verify SegFormer import no longer poisons other trainers
python -c "
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSegFormer import nnUNetTrainerSegFormer
print('SegFormer import OK')
"

# 2. Verify all trainers can be discovered
python -c "
import os, importlib, pkgutil
import nnunetv2
base = os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer')
pkg = 'nnunetv2.training.nnUNetTrainer'
failed = []
for m in pkgutil.iter_modules([base]):
    try:
        importlib.import_module(pkg + '.' + m.name)
        print(f'[OK] {m.name}')
    except Exception as e:
        print(f'[FAIL] {m.name}: {type(e).__name__}: {e}')
        failed.append(m.name)
print(f'\nResults: {len(failed)} failed / total')
"

# 3. Verify BaseChainLMambaMFDS can be instantiated
python -c "
from nnunetv2.run.run_training import get_trainer_from_args
trainer = get_trainer_from_args(305, '3d_fullres', 0, 'nnUNetTrainerBaseChainLMambaMFDS')
print('BaseChain init OK')
print('use_lmamba:', trainer._use_lmamba)
"

# 4. Verify ConfigurationManager access no longer fails
python -c "
from nnunetv2.run.run_training import get_trainer_from_args
trainer = get_trainer_from_args(305, '3d_fullres', 0, 'nnUNetTrainer')
print('Base trainer init OK')
print('Features:', trainer.configuration_manager.configuration.get('UNet_base_num_features', 'N/A'))
"
```

---

## 6. Root Cause Summary

**Why "running A model shows errors for B model"**: nnU-Net's `recursive_find_python_class()` enumerates and imports ALL trainer modules to discover the requested class. A single broken `import` statement in ONE file (SegFormer) causes the ENTIRE trainer directory to become unusable. This is why running `-tr nnUNetTrainerSegMamba` shows errors about `RobustSegFormer3D` — SegMamba was never reached; the discovery process died at SegFormer.
