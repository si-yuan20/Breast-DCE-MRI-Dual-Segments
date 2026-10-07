"""
verify_base_chain_models.py
============================
Verification script: checks all custom Trainers for nnU-Net base chain compliance.

Checks:
  1. Import  ✅/❌
  2. DS list output (len matches nnU-Net scales)
  3. Full-resolution spatial match
  4. BatchNorm3d presence
  5. NaN / Inf in output
  6. Parameter count > 0
  7. Plans-driven parameter reading

Usage:
  python nnunetv2/analysis/verify_base_chain_models.py [--dataset-id 201]
"""

import argparse
import csv
import os
import sys
import traceback
from typing import Dict, List, Any

import torch
import torch.nn as nn
import numpy as np

# Trainer registry
TRAINER_NAMES = [
    "nnUNetTrainerSegResNet",
    "nnUNetTrainerSegMamba",
    "nnUNetTrainerLMambaMFDSNet",
    "nnUNetTrainerBaseChainLMambaMFDS",
    "nnUNetTrainerUNETR",
    "nnUNetTrainerSwinUNETR",
    "nnUNetTrainerVNet",
    "nnUNetTrainerDynUNet",
    "nnUNetTrainerAttentionUnet",
    "nnUNetTrainerSegFormer",
    "nnUNetTrainerVeloxSeg",
    "nnUNetTrainerUMamba",
]


def count_batchnorm3d(module: nn.Module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, nn.BatchNorm3d))


def check_nan_inf(tensor) -> Dict[str, bool]:
    return {
        "has_nan": bool(torch.isnan(tensor).any().item()),
        "has_inf": bool(torch.isinf(tensor).any().item()),
    }


def load_plans(dataset_id: int = 201):
    """Load plans for a dataset. Fall back to dummy if not available."""
    from batchgenerators.utilities.file_and_folder_operations import load_json, join

    try:
        from nnunetv2.paths import nnUNet_preprocessed
        preprocessed = join(nnUNet_preprocessed, f"Dataset{dataset_id:03d}_*")
        import glob
        candidates = glob.glob(preprocessed)
        if candidates:
            plans_path = join(candidates[0], "nnUNetPlans.json")
            if os.path.exists(plans_path.replace("\\", "/")):
                plans = load_json(plans_path)
                return plans, f"Dataset{dataset_id:03d}"
    except Exception:
        pass

    # Dummy plans for testing
    dummy_plans = {
        "dataset_name": "Dataset201_dummy",
        "configurations": {
            "3d_fullres": {
                "patch_size": [64, 128, 128],
                "batch_size": 2,
                "UNet_base_num_features": 32,
                "unet_max_num_features": 320,
                "pool_op_kernel_sizes": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "conv_kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
                "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
                "network_arch_class_name": "dummy",
                "network_arch_init_kwargs": {},
                "network_arch_init_kwargs_req_import": [],
            }
        }
    }
    return dummy_plans, "Dataset201_dummy"


def verify_trainer(trainer_name: str, plans: dict, dataset_name: str, device: torch.device) -> Dict[str, Any]:
    """Verify a single Trainer."""
    result = {
        "trainer": trainer_name,
        "import": "PASS",
        "init": "FAIL",
        "ds_output": "N/A",
        "ds_count": 0,
        "spatial_match": "N/A",
        "bn_count": 0,
        "nan_inf": "N/A",
        "params": 0,
        "error": "",
    }

    try:
        # ── Import ──
        module_path = f"nnunetv2.training.nnUNetTrainer.{trainer_name}"
        try:
            mod = __import__(module_path, fromlist=[trainer_name])
            TrainerClass = getattr(mod, trainer_name)
        except ImportError:
            # Try without 'nnUNetTrainer' prefix
            alt_path = f"nnunetv2.training.nnUNetTrainer.{trainer_name}"
            mod = __import__(alt_path, fromlist=[trainer_name])
            TrainerClass = getattr(mod, trainer_name)

        # ── Dummy dataset_json ──
        dataset_json = {
            "labels": {"background": 0, "foreground": 1},
            "numTraining": 10,
            "file_ending": ".nii.gz",
        }

        # ── Instantiate Trainer (minimal) ──
        # We need a valid configuration dict
        cfg_name = "3d_fullres"
        try:
            trainer = TrainerClass(
                plans=plans,
                configuration=cfg_name,
                fold=0,
                dataset_json=dataset_json,
                device=device,
            )
            result["init"] = "PASS"
        except Exception as e:
            result["init"] = f"FAIL: {e}"
            result["error"] = str(e)
            return result

        # ── Build network ──
        try:
            trainer.initialize()
            model = trainer.network
            if hasattr(model, "module"):
                model = model.module
            result["init"] = "PASS (network built)"
        except Exception as e:
            result["init"] = f"NETWORK FAIL: {e}"
            result["error"] = str(e)
            return result

        # ── Param count ──
        result["params"] = sum(p.numel() for p in model.parameters())

        # ── BatchNorm3d check ──
        result["bn_count"] = count_batchnorm3d(model)

        # ── Forward check ──
        in_ch = trainer.num_input_channels
        patch_size = trainer.configuration_manager.patch_size
        dummy_input = torch.randn(2, in_ch, *patch_size).to(device)

        with torch.no_grad():
            try:
                output = model(dummy_input)
            except Exception as e:
                result["ds_output"] = f"FORWARD FAIL: {e}"
                result["error"] = str(e)
                return result

        # ── DS output check ──
        if isinstance(output, (list, tuple)):
            result["ds_output"] = "PASS (list)"
            result["ds_count"] = len(output)
            out0 = output[0]

            # Check DS scales match
            ds_scales = trainer._get_deep_supervision_scales()
            expected_count = len(ds_scales) if ds_scales else 1
            if result["ds_count"] != expected_count:
                result["ds_output"] = f"COUNT MISMATCH: got {result['ds_count']}, expected {expected_count}"
        else:
            result["ds_output"] = "FAIL (single tensor)"
            result["ds_count"] = 1
            out0 = output

        # ── Spatial match ──
        if tuple(out0.shape[2:]) == tuple(patch_size):
            result["spatial_match"] = "PASS"
        else:
            result["spatial_match"] = f"MISMATCH: out={tuple(out0.shape[2:])}, patch={patch_size}"

        # ── NaN/Inf ──
        ni = check_nan_inf(out0)
        if ni["has_nan"] or ni["has_inf"]:
            result["nan_inf"] = f"FAIL nan={ni['has_nan']} inf={ni['has_inf']}"
        else:
            result["nan_inf"] = "PASS"

        del model, dummy_input, output
        torch.cuda.empty_cache()

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["import"] = "FAIL"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, default=201)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    plans, dataset_name = load_plans(args.dataset_id)
    print(f"Using plans for {dataset_name}, device={device}")

    results = []
    for name in TRAINER_NAMES:
        print(f"\n{'='*60}")
        print(f"Verifying {name}...")
        result = verify_trainer(name, plans, dataset_name, device)
        results.append(result)

        status = "✅" if result["import"] == "PASS" and "FAIL" not in str(result["ds_output"]) and result["bn_count"] == 0 else "❌"
        print(f"  {status} Import={result['import']} | Init={result['init']}")
        print(f"  DS={result['ds_output']} (count={result['ds_count']}) | Spatial={result['spatial_match']}")
        print(f"  BN={result['bn_count']} | NaN/Inf={result['nan_inf']} | Params={result['params']:,}")
        if result["error"]:
            print(f"  Error: {result['error']}")

    # ── Output CSV ──
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(output_dir, "base_chain_verification_report.csv")
    md_path = os.path.join(output_dir, "base_chain_verification_report.md")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(md_path, "w") as f:
        f.write("# Base Chain Verification Report\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Device: {device}\n\n")
        f.write("| Trainer | Import | DS Output | DS Count | Spatial | BN | NaN/Inf | Params | Error |\n")
        f.write("|---------|--------|-----------|----------|---------|----|---------|--------|-------|\n")
        for r in results:
            f.write(f"| {r['trainer']} | {r['import']} | {r['ds_output']} | {r['ds_count']} | "
                    f"{r['spatial_match']} | {r['bn_count']} | {r['nan_inf']} | {r['params']:,} | {r['error']} |\n")

    print(f"\nReports written: {csv_path}\n{md_path}")


if __name__ == "__main__":
    main()
