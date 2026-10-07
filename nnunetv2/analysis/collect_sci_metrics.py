"""
collect_sci_metrics.py
======================
Aggregate per-fold metrics from nnUNet_results into SCI-paper-ready tables.

Outputs:
  results_summary.csv     — mean ± std across folds per model
  per_case_metrics.csv    — per-case Dice/HD95/ASSD for box plots
  results_summary.xlsx    — formatted Excel

Usage:
  python nnunetv2/analysis/collect_sci_metrics.py --dataset-id 201 \
      --models nnUNetTrainer nnUNetTrainerSegResNet nnUNetTrainerSegMamba \
      --metrics Dice HD95 ASSD
"""

import argparse
import csv
import json
import os
import glob
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd


def find_fold_dirs(nnunet_results: str, dataset_name: str, trainer_name: str) -> List[str]:
    """Find all fold directories for a given trainer."""
    pattern = os.path.join(
        nnunet_results, dataset_name,
        f"{trainer_name}__*", "fold_*"
    )
    # Replace backslashes for cross-platform
    pattern = pattern.replace("\\", "/")
    dirs = sorted(glob.glob(pattern))
    fold_dirs = []
    for d in dirs:
        if os.path.isdir(d):
            summary = os.path.join(d, "validation", "summary.json")
            if os.path.exists(summary):
                fold_dirs.append(d)
    return fold_dirs


def read_fold_metrics(fold_dir: str, foreground_labels: List[int]) -> Dict[str, Any]:
    """Read metrics from a single fold's validation/summary.json."""
    summary_path = os.path.join(fold_dir, "validation", "summary.json")
    if not os.path.exists(summary_path):
        return {}

    with open(summary_path, "r") as f:
        summary = json.load(f)

    result = {}
    # Foreground mean
    if "foreground_mean" in summary:
        for metric in summary["foreground_mean"]:
            result[f"foreground_{metric}"] = summary["foreground_mean"][metric]

    # Per-class mean
    if "mean" in summary:
        for label, metrics in summary["mean"].items():
            for metric_name, value in metrics.items():
                result[f"class_{label}_{metric_name}"] = value

    # Overall mean Dice
    if "mean" in summary:
        dices = [m.get("Dice", 0) for _, m in summary["mean"].items()]
        if dices:
            result["mean_dice"] = np.mean(dices)

    return result


def read_per_case_metrics(fold_dir: str) -> List[Dict]:
    """Read per-case metrics if available."""
    per_case_path = os.path.join(fold_dir, "validation", "per_case.json")
    if os.path.exists(per_case_path):
        with open(per_case_path, "r") as f:
            return json.load(f)
    return []


def aggregate_folds(
    nnunet_results: str,
    dataset_name: str,
    trainer_names: List[str],
    foreground_labels: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate metrics across folds for all trainers."""
    summary_rows = []
    per_case_rows = []

    for trainer_name in trainer_names:
        fold_dirs = find_fold_dirs(nnunet_results, dataset_name, trainer_name)
        if not fold_dirs:
            print(f"WARNING: No results found for {trainer_name} in {dataset_name}")
            continue

        all_metrics: List[Dict] = []
        for fd in fold_dirs:
            metrics = read_fold_metrics(fd, foreground_labels)
            metrics["fold"] = os.path.basename(fd)
            all_metrics.append(metrics)

            # Per-case
            cases = read_per_case_metrics(fd)
            for c in cases:
                c["trainer"] = trainer_name
                c["fold"] = os.path.basename(fd)
                per_case_rows.append(c)

        if not all_metrics:
            continue

        # Compute mean ± std across folds
        metric_names = set()
        for m in all_metrics:
            metric_names.update(k for k in m.keys() if k != "fold")

        row = {"trainer": trainer_name, "n_folds": len(fold_dirs)}
        for name in sorted(metric_names):
            values = [m.get(name, np.nan) for m in all_metrics]
            values = [v for v in values if not np.isnan(v)]
            if values:
                row[name] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                row[f"{name}_mean"] = np.mean(values)
                row[f"{name}_std"] = np.std(values)
            else:
                row[name] = "N/A"

        # Count params (estimate from summary.json metadata if available)
        summary_rows.append(row)
        print(f"  {trainer_name}: {len(fold_dirs)} folds, metrics={len(metric_names)}")

    df_summary = pd.DataFrame(summary_rows)
    df_per_case = pd.DataFrame(per_case_rows) if per_case_rows else pd.DataFrame()

    return df_summary, df_per_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--foreground-labels", nargs="+", type=int, default=[1])
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    from nnunetv2.paths import nnUNet_results
    dataset_name = f"Dataset{args.dataset_id:03d}"

    if not args.models:
        # Auto-discover trainers from nnUNet_results
        pattern = os.path.join(nnUNet_results, dataset_name, "*__*")
        pattern = pattern.replace("\\", "/")
        dirs = glob.glob(pattern)
        trainer_names = list(set(
            os.path.basename(d).split("__")[0] for d in dirs
            if os.path.isdir(d)
        ))
        print(f"Auto-discovered {len(trainer_names)} trainers")
    else:
        trainer_names = args.models

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))

    df_summary, df_per_case = aggregate_folds(
        nnUNet_results, dataset_name, trainer_names, args.foreground_labels,
    )

    # Save
    csv_path = os.path.join(output_dir, "results_summary.csv")
    xlsx_path = os.path.join(output_dir, "results_summary.xlsx")
    per_case_path = os.path.join(output_dir, "per_case_metrics.csv")

    df_summary.to_csv(csv_path, index=False)
    print(f"Summary: {csv_path} ({len(df_summary)} models)")

    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            if not df_per_case.empty:
                df_per_case.to_excel(writer, sheet_name="PerCase", index=False)
        print(f"Excel: {xlsx_path}")
    except Exception:
        pass

    if not df_per_case.empty:
        df_per_case.to_csv(per_case_path, index=False)
        print(f"Per-case: {per_case_path} ({len(df_per_case)} cases)")

    # Print summary
    print("\n=== Results Summary ===")
    for _, row in df_summary.iterrows():
        print(f"{row['trainer']}: {row.get('foreground_Dice', 'N/A')}")

    print("\nDone. Run statistical_test.py for significance testing.")


if __name__ == "__main__":
    main()
