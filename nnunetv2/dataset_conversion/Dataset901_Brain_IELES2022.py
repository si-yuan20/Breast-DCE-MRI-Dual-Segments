#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset901_Brain_ISLES2022.py

Convert ISLES 2022 stroke lesion segmentation dataset
to nnU-Net v2 raw dataset format.

Source structure:

/data/sdb/medical/datasets/brain/ISLES2022/ISLES-2022/
├── sub-strokecase0001/
│   └── ses-0001/
│       └── dwi/
│           ├── sub-strokecase0001_ses-0001_dwi.nii.gz
│           ├── sub-strokecase0001_ses-0001_adc.nii.gz
│           └── ...
├── sub-strokecase0002/
│   └── ...
└── derivatives/
    ├── sub-strokecase0001/
    │   └── ses-0001/
    │       └── sub-strokecase0001_ses-0001_msk.nii.gz
    └── ...

Output:

/data/sdb/medical/nnunet_data/nnUNet_raw/Dataset901_Brain_ISLES2022/
├── imagesTr/
│   ├── ISLES_strokecase0001_ses0001_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── ISLES_strokecase0001_ses0001.nii.gz
│   └── ...
└── dataset.json

Default:
    channel 0000 = DWI

Optional:
    --channels dwi_adc

Then:
    channel 0000 = DWI
    channel 0001 = ADC

Important:
    This script DOES NOT:
    - resample
    - register
    - flip left/right
    - modify affine
    - normalize intensity

It only reorganizes the dataset into nnU-Net v2 format
and standardizes segmentation labels to uint8 {0, 1}.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import nibabel as nib


# ============================================================
# Default paths
# ============================================================

DEFAULT_SOURCE_ROOT = Path(
    "/data/sdb/medical/datasets/brain/ISLES2022/ISLES-2022"
)

DEFAULT_TARGET_ROOT = Path(
    "/data/sdb/medical/nnunet_data/nnUNet_raw/"
    "Dataset901_Brain_ISLES2022"
)


# ============================================================
# Configuration
# ============================================================

# Maximum accepted distance from an integer segmentation value.
#
# Examples:
#
# 0.999999999767 -> accepted and converted to 1
# 0.000000000012 -> accepted and converted to 0
#
# But something such as 0.43 will fail.
LABEL_INTEGER_TOLERANCE = 1e-3


# ============================================================
# Helper functions
# ============================================================

def sanitize_case_identifier(text: str) -> str:
    """
    Make case identifier safe for nnU-Net filenames.
    """

    text = text.strip()

    text = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    return text.strip("_")


def build_case_id(
    subject: str,
    session: str,
) -> str:
    """
    Example:

    subject:
        sub-strokecase0001

    session:
        ses-0001

    output:
        ISLES_strokecase0001_ses0001
    """

    subject_clean = subject

    if subject_clean.startswith("sub-"):
        subject_clean = subject_clean[4:]

    subject_clean = subject_clean.replace("-", "")
    session_clean = session.replace("-", "")

    return sanitize_case_identifier(
        f"ISLES_{subject_clean}_{session_clean}"
    )


def load_nifti(path: Path):
    """
    Load NIfTI safely.
    """

    try:
        return nib.load(str(path))

    except Exception as exc:

        raise RuntimeError(
            f"\nCannot read NIfTI file:\n"
            f"{path}\n"
            f"Error: {exc}"
        ) from exc


# ============================================================
# Geometry validation
# ============================================================

def validate_geometry(
    reference_path: Path,
    other_path: Path,
    description: str,
):
    """
    Validate image geometry.

    Shape and affine must match.

    No automatic resampling is performed.
    """

    ref = load_nifti(reference_path)
    other = load_nifti(other_path)

    if ref.shape != other.shape:

        raise RuntimeError(
            "\n"
            + "=" * 80
            + "\nGEOMETRY ERROR\n"
            + "=" * 80
            + "\n"
            f"{description}\n\n"
            f"Reference:\n"
            f"{reference_path}\n"
            f"Shape: {ref.shape}\n\n"
            f"Other:\n"
            f"{other_path}\n"
            f"Shape: {other.shape}\n"
        )

    if not np.allclose(
        ref.affine,
        other.affine,
        rtol=1e-5,
        atol=1e-4,
    ):

        raise RuntimeError(
            "\n"
            + "=" * 80
            + "\nAFFINE ERROR\n"
            + "=" * 80
            + "\n"
            f"{description}\n\n"
            f"Reference:\n{reference_path}\n\n"
            f"Reference affine:\n"
            f"{ref.affine}\n\n"
            f"Other:\n{other_path}\n\n"
            f"Other affine:\n"
            f"{other.affine}\n\n"
            "The converter intentionally does not resample "
            "or register the mask automatically."
        )


# ============================================================
# Label validation and normalization
# ============================================================

def inspect_label(
    label_path: Path,
):
    """
    Validate lesion mask.

    Accepted examples:

        0
        1

        0.0
        1.0

        0.9999999997671694
        0.000000000001

    These floating point values will later be rounded and saved
    as uint8 {0, 1}.

    Unexpected values such as:
        0.5
        2
        255

    cause conversion to stop.
    """

    img = load_nifti(label_path)

    data = np.asanyarray(
        img.dataobj
    )

    if data.size == 0:

        raise RuntimeError(
            f"Empty segmentation:\n{label_path}"
        )

    if not np.all(np.isfinite(data)):

        raise RuntimeError(
            f"Segmentation contains NaN or Inf:\n"
            f"{label_path}"
        )

    rounded = np.rint(data)

    max_error = float(
        np.max(
            np.abs(
                data.astype(np.float64)
                - rounded.astype(np.float64)
            )
        )
    )

    if max_error > LABEL_INTEGER_TOLERANCE:

        values = np.unique(data)

        if len(values) > 30:
            value_text = (
                f"{values[:30].tolist()} "
                f"... total unique={len(values)}"
            )
        else:
            value_text = values.tolist()

        raise RuntimeError(
            "\n"
            + "=" * 80
            + "\nINVALID SEGMENTATION VALUES\n"
            + "=" * 80
            + "\n"
            f"File:\n{label_path}\n\n"
            f"Values:\n{value_text}\n\n"
            f"Maximum distance from integer: "
            f"{max_error}\n\n"
            f"Tolerance: "
            f"{LABEL_INTEGER_TOLERANCE}\n\n"
            "This does not look like a binary integer mask."
        )

    rounded_values = np.unique(
        rounded
    )

    if not np.all(
        np.isin(
            rounded_values,
            [0, 1],
        )
    ):

        raise RuntimeError(
            "\n"
            + "=" * 80
            + "\nINVALID LABEL CLASSES\n"
            + "=" * 80
            + "\n"
            f"File:\n{label_path}\n\n"
            f"Rounded classes:\n"
            f"{rounded_values.tolist()}\n\n"
            "Expected classes:\n"
            "[0, 1]"
        )

    return {
        "unique_original": np.unique(data),
        "unique_rounded": rounded_values,
        "max_rounding_error": max_error,
    }


def save_normalized_binary_label(
    src: Path,
    dst: Path,
):
    """
    Save segmentation as true uint8 {0, 1}.

    Preserves:
    - affine
    - qform
    - sform
    - qform code
    - sform code

    It does NOT modify geometry.
    """

    img = load_nifti(src)

    data = np.asanyarray(
        img.dataobj
    )

    rounded = np.rint(data)

    binary = rounded.astype(
        np.uint8
    )

    if not np.all(
        np.isin(
            np.unique(binary),
            [0, 1],
        )
    ):

        raise RuntimeError(
            f"Normalized mask unexpectedly "
            f"contains non-binary values:\n{src}"
        )

    # Copy header instead of modifying original one.
    new_header = img.header.copy()

    new_header.set_data_dtype(
        np.uint8
    )

    new_img = nib.Nifti1Image(
        binary,
        img.affine,
        header=new_header,
    )

    # Preserve qform / sform explicitly.
    qform, qform_code = img.get_qform(
        coded=True
    )

    sform, sform_code = img.get_sform(
        coded=True
    )

    if qform is not None:
        new_img.set_qform(
            qform,
            int(qform_code),
        )

    if sform is not None:
        new_img.set_sform(
            sform,
            int(sform_code),
        )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        new_img,
        str(dst),
    )


# ============================================================
# File operations
# ============================================================

def copy_nifti(
    src: Path,
    dst: Path,
):
    """
    Copy input MRI without modifying it.
    """

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        src,
        dst,
    )


def prepare_output_directory(
    target_root: Path,
    clean: bool,
):
    """
    Prepare Dataset901 output.

    --clean ONLY clears this exact dataset folder contents.
    """

    images_tr = target_root / "imagesTr"
    labels_tr = target_root / "labelsTr"
    images_ts = target_root / "imagesTs"
    dataset_json = target_root / "dataset.json"

    target_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = []

    for directory in [
        images_tr,
        labels_tr,
        images_ts,
    ]:

        if (
            directory.exists()
            and any(directory.iterdir())
        ):
            existing.append(
                str(directory)
            )

    if dataset_json.exists():
        existing.append(
            str(dataset_json)
        )

    if existing and not clean:

        raise RuntimeError(
            "\nTarget dataset already contains data:\n\n"
            + "\n".join(existing)
            + "\n\nUse --clean if you want to rebuild Dataset901."
        )

    if clean:

        for directory in [
            images_tr,
            labels_tr,
            images_ts,
        ]:

            if directory.exists():

                shutil.rmtree(
                    directory
                )

        if dataset_json.exists():

            dataset_json.unlink()

    images_tr.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_tr.mkdir(
        parents=True,
        exist_ok=True,
    )

    return images_tr, labels_tr


# ============================================================
# Case discovery
# ============================================================

def discover_isles_cases(
    source_root: Path,
    channels: str,
) -> List[Dict]:
    """
    Discover complete ISLES image-mask pairs.
    """

    if not source_root.exists():

        raise FileNotFoundError(
            f"ISLES source root does not exist:\n"
            f"{source_root}"
        )

    derivatives_root = (
        source_root
        / "derivatives"
    )

    if not derivatives_root.exists():

        raise FileNotFoundError(
            f"ISLES derivatives directory does not exist:\n"
            f"{derivatives_root}"
        )

    dwi_files = sorted(
        source_root.glob(
            "sub-strokecase*/ses-*/dwi/*_dwi.nii.gz"
        )
    )

    if not dwi_files:

        raise RuntimeError(
            "No ISLES DWI files found.\n\n"
            f"Source:\n{source_root}\n\n"
            "Expected pattern:\n"
            "sub-strokecase*/ses-*/dwi/*_dwi.nii.gz"
        )

    print(
        f"[INFO] Found {len(dwi_files)} candidate DWI files."
    )

    cases = []

    used_case_ids = set()

    missing = []

    for dwi_candidate in dwi_files:

        dwi_dir = dwi_candidate.parent
        session_dir = dwi_dir.parent
        subject_dir = session_dir.parent

        subject = subject_dir.name
        session = session_dir.name

        case_id = build_case_id(
            subject,
            session,
        )

        if case_id in used_case_ids:

            raise RuntimeError(
                f"Duplicate case ID detected:\n"
                f"{case_id}"
            )

        used_case_ids.add(
            case_id
        )

        dwi_path = (
            dwi_dir
            / f"{subject}_{session}_dwi.nii.gz"
        )

        adc_path = (
            dwi_dir
            / f"{subject}_{session}_adc.nii.gz"
        )

        mask_path = (
            derivatives_root
            / subject
            / session
            / f"{subject}_{session}_msk.nii.gz"
        )

        if not dwi_path.exists():

            missing.append(
                str(dwi_path)
            )

            continue

        if not mask_path.exists():

            missing.append(
                str(mask_path)
            )

            continue

        if (
            channels == "dwi_adc"
            and not adc_path.exists()
        ):

            missing.append(
                str(adc_path)
            )

            continue

        case = {
            "case_id": case_id,
            "subject": subject,
            "session": session,
            "dwi": dwi_path,
            "mask": mask_path,
        }

        if channels == "dwi_adc":

            case["adc"] = adc_path

        cases.append(
            case
        )

    if missing:

        print(
            "\n[ERROR] Missing required files:"
        )

        for path in missing:

            print(
                f"  - {path}"
            )

        raise RuntimeError(
            f"\nMissing {len(missing)} required files."
        )

    if not cases:

        raise RuntimeError(
            "No complete ISLES cases found."
        )

    return cases


# ============================================================
# Source validation
# ============================================================

def validate_cases(
    cases: List[Dict],
    channels: str,
):
    """
    Validate all cases before target data is written.
    """

    print(
        "\n[INFO] Validating source dataset..."
    )

    total = len(cases)

    maximum_rounding_error = 0.0
    floating_masks = 0

    for index, case in enumerate(
        cases,
        start=1,
    ):

        case_id = case["case_id"]

        dwi_path = Path(
            case["dwi"]
        )

        mask_path = Path(
            case["mask"]
        )

        validate_geometry(
            dwi_path,
            mask_path,
            f"{case_id}: DWI vs lesion mask",
        )

        if channels == "dwi_adc":

            adc_path = Path(
                case["adc"]
            )

            validate_geometry(
                dwi_path,
                adc_path,
                f"{case_id}: DWI vs ADC",
            )

        label_info = inspect_label(
            mask_path
        )

        error = label_info[
            "max_rounding_error"
        ]

        maximum_rounding_error = max(
            maximum_rounding_error,
            error,
        )

        if error > 0:

            floating_masks += 1

        if (
            index == 1
            or index % 25 == 0
            or index == total
        ):

            print(
                f"[VALIDATE] "
                f"{index:4d}/{total:4d} "
                f"{case_id}"
            )

    print(
        "[OK] All ISLES source cases passed validation."
    )

    print(
        f"[INFO] Masks with floating-point "
        f"rounding error: {floating_masks}"
    )

    print(
        f"[INFO] Maximum observed rounding error: "
        f"{maximum_rounding_error:.12g}"
    )


# ============================================================
# dataset.json
# ============================================================

def create_dataset_json(
    target_root: Path,
    num_training: int,
    channels: str,
):
    """
    Generate nnU-Net v2 dataset.json.
    """

    if channels == "dwi":

        channel_names = {
            "0": "DWI"
        }

    elif channels == "dwi_adc":

        channel_names = {
            "0": "DWI",
            "1": "ADC",
        }

    else:

        raise ValueError(
            f"Unsupported channels: {channels}"
        )

    dataset = {
        "name": "Brain_ISLES2022",
        "description": (
            "ISLES 2022 ischemic stroke lesion segmentation"
        ),
        "channel_names": channel_names,
        "labels": {
            "background": 0,
            "stroke_lesion": 1,
        },
        "numTraining": int(
            num_training
        ),
        "file_ending": ".nii.gz",
    }

    json_path = (
        target_root
        / "dataset.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            dataset,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"[OK] dataset.json written:\n"
        f"{json_path}"
    )


# ============================================================
# Final output validation
# ============================================================

def validate_output(
    target_root: Path,
    cases: List[Dict],
    channels: str,
):
    """
    Validate generated nnU-Net raw dataset.
    """

    images_tr = (
        target_root
        / "imagesTr"
    )

    labels_tr = (
        target_root
        / "labelsTr"
    )

    n_channels = (
        1
        if channels == "dwi"
        else 2
    )

    expected_images = (
        len(cases)
        * n_channels
    )

    image_files = list(
        images_tr.glob(
            "*.nii.gz"
        )
    )

    label_files = list(
        labels_tr.glob(
            "*.nii.gz"
        )
    )

    if len(image_files) != expected_images:

        raise RuntimeError(
            "Image count mismatch.\n"
            f"Expected: {expected_images}\n"
            f"Actual: {len(image_files)}"
        )

    if len(label_files) != len(cases):

        raise RuntimeError(
            "Label count mismatch.\n"
            f"Expected: {len(cases)}\n"
            f"Actual: {len(label_files)}"
        )

    print(
        "\n[INFO] Verifying normalized output masks..."
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):

        case_id = case[
            "case_id"
        ]

        output_label = (
            labels_tr
            / f"{case_id}.nii.gz"
        )

        img = nib.load(
            str(output_label)
        )

        data = np.asanyarray(
            img.dataobj
        )

        values = np.unique(
            data
        )

        if not np.all(
            np.isin(
                values,
                [0, 1],
            )
        ):

            raise RuntimeError(
                f"Output mask is not binary:\n"
                f"{output_label}\n"
                f"Values: {values.tolist()}"
            )

        if data.dtype != np.uint8:

            # nibabel may expose proxy dtype differently;
            # header check below is more authoritative.
            dtype_header = (
                img.header.get_data_dtype()
            )

            if dtype_header != np.dtype(
                np.uint8
            ):

                raise RuntimeError(
                    f"Output label is not uint8:\n"
                    f"{output_label}\n"
                    f"dtype={dtype_header}"
                )

        if (
            index == 1
            or index % 50 == 0
            or index == len(cases)
        ):

            print(
                f"[OUTPUT CHECK] "
                f"{index:4d}/{len(cases):4d}"
            )

    print(
        "[OK] Output dataset validation passed."
    )


# ============================================================
# Conversion
# ============================================================

def convert(
    source_root: Path,
    target_root: Path,
    channels: str,
    clean: bool,
):
    """
    Main conversion pipeline.
    """

    print(
        "=" * 80
    )

    print(
        "ISLES 2022 -> nnU-Net v2 conversion"
    )

    print(
        "=" * 80
    )

    print(
        f"Source : {source_root}"
    )

    print(
        f"Target : {target_root}"
    )

    print(
        f"Mode   : {channels}"
    )

    # --------------------------------------------------------
    # Step 1
    # Discover
    # --------------------------------------------------------

    cases = discover_isles_cases(
        source_root,
        channels,
    )

    print(
        f"[INFO] Valid paired cases: "
        f"{len(cases)}"
    )

    # --------------------------------------------------------
    # Step 2
    # Validate ALL source cases
    # --------------------------------------------------------

    validate_cases(
        cases,
        channels,
    )

    # --------------------------------------------------------
    # Step 3
    # Prepare output directory
    # --------------------------------------------------------

    images_tr, labels_tr = (
        prepare_output_directory(
            target_root,
            clean,
        )
    )

    # --------------------------------------------------------
    # Step 4
    # Convert/copy
    # --------------------------------------------------------

    print(
        "\n[INFO] Writing nnU-Net dataset..."
    )

    total = len(cases)

    for index, case in enumerate(
        cases,
        start=1,
    ):

        case_id = (
            case["case_id"]
        )

        dwi_src = Path(
            case["dwi"]
        )

        mask_src = Path(
            case["mask"]
        )

        dwi_dst = (
            images_tr
            / f"{case_id}_0000.nii.gz"
        )

        mask_dst = (
            labels_tr
            / f"{case_id}.nii.gz"
        )

        # MRI is copied exactly.
        copy_nifti(
            dwi_src,
            dwi_dst,
        )

        # Mask is standardized to true uint8 0/1.
        save_normalized_binary_label(
            mask_src,
            mask_dst,
        )

        if channels == "dwi_adc":

            adc_src = Path(
                case["adc"]
            )

            adc_dst = (
                images_tr
                / f"{case_id}_0001.nii.gz"
            )

            copy_nifti(
                adc_src,
                adc_dst,
            )

        if (
            index == 1
            or index % 25 == 0
            or index == total
        ):

            print(
                f"[WRITE] "
                f"{index:4d}/{total:4d} "
                f"{case_id}"
            )

    # --------------------------------------------------------
    # Step 5
    # dataset.json
    # --------------------------------------------------------

    create_dataset_json(
        target_root,
        len(cases),
        channels,
    )

    # --------------------------------------------------------
    # Step 6
    # Final validation
    # --------------------------------------------------------

    validate_output(
        target_root,
        cases,
        channels,
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    channel_count = (
        1
        if channels == "dwi"
        else 2
    )

    print()
    print(
        "=" * 80
    )

    print(
        "[SUCCESS] ISLES 2022 conversion completed"
    )

    print(
        "=" * 80
    )

    print(
        f"Cases      : {len(cases)}"
    )

    print(
        f"Channels   : {channel_count}"
    )

    print(
        f"Images     : "
        f"{len(cases) * channel_count}"
    )

    print(
        f"Labels     : {len(cases)}"
    )

    print(
        f"Output     : {target_root}"
    )

    print()

    print(
        "Next:"
    )

    print(
        "nnUNetv2_plan_and_preprocess "
        "-d 901 --verify_dataset_integrity"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Convert ISLES 2022 "
            "to nnU-Net v2 raw format."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )

    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
    )

    parser.add_argument(
        "--channels",
        choices=[
            "dwi",
            "dwi_adc",
        ],
        default="dwi",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Rebuild Dataset901 by removing "
            "previous generated files."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    try:

        convert(
            source_root=args.source.resolve(),
            target_root=args.target.resolve(),
            channels=args.channels,
            clean=args.clean,
        )

    except Exception as exc:

        print()

        print(
            "=" * 80
        )

        print(
            "[FAILED] ISLES conversion failed"
        )

        print(
            "=" * 80
        )

        print(
            str(exc)
        )

        sys.exit(1)


if __name__ == "__main__":
    main()