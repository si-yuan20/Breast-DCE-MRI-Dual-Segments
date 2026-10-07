#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset902_Brain_Atlas2.py

Convert ATLAS preprocessed stroke lesion dataset
to nnU-Net v2 raw dataset format.

Source
------
/data/sdb/medical/datasets/brain/ATLAS/ATLAS3_Training_Preprocessed/

Example:

R001/
└── sub-r001s001/
    └── ses-1/
        └── anat/
            ├── sub-r001s001_ses-1_metadata.csv
            ├── sub-r001s001_ses-1_space-MNI152NLin2009aSym_desc-brainnorm_T1w.nii.gz
            └── sub-r001s001_ses-1_space-MNI152NLin2009aSym_label-lesion_desc-T1lesion_mask.nii.gz


Output
------
/data/sdb/medical/nnunet_data/nnUNet_raw/Dataset902_Brain_Atlas2/

├── imagesTr/
│   ├── ATLAS_r001s001_ses1_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── ATLAS_r001s001_ses1.nii.gz
│   └── ...
├── dataset.json
└── excluded_cases.json


Channel
-------
0000 = T1


Label
-----
0 = background
1 = stroke lesion


Important behavior
------------------
1. Empty T1/mask files are excluded.
2. Corrupted/unreadable T1/mask files are excluded.
3. Missing masks are excluded.
4. Excluded cases are recorded in excluded_cases.json.
5. Floating-point binary masks such as:
       0.999999999767 -> 1
   are converted safely to uint8 0/1.
6. Geometry mismatch is treated as a REAL data error and stops conversion.
7. Non-binary segmentation classes are treated as a REAL data error.
8. No image resampling, registration, flipping, reorientation,
   or intensity preprocessing is performed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


# ============================================================
# Default paths
# ============================================================

DEFAULT_SOURCE_ROOT = Path(
    "/data/sdb/medical/datasets/brain/ATLAS/"
    "ATLAS3_Training_Preprocessed"
)

DEFAULT_TARGET_ROOT = Path(
    "/data/sdb/medical/nnunet_data/nnUNet_raw/"
    "Dataset902_Brain_Atlas2"
)


# ============================================================
# Configuration
# ============================================================

LABEL_INTEGER_TOLERANCE = 1e-3

# A NIfTI must contain more than zero bytes.
MIN_FILE_SIZE_BYTES = 1


# ============================================================
# Basic helpers
# ============================================================

def sanitize_case_identifier(text: str) -> str:
    text = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        text.strip(),
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

    subject_clean = subject

    if subject_clean.startswith("sub-"):
        subject_clean = subject_clean[4:]

    subject_clean = subject_clean.replace("-", "")
    session_clean = session.replace("-", "")

    return sanitize_case_identifier(
        f"ATLAS_{subject_clean}_{session_clean}"
    )


# ============================================================
# File checks
# ============================================================

def check_file_basic(
    path: Path,
) -> Tuple[bool, Optional[str]]:
    """
    Check:
    - exists
    - is file
    - non-zero size
    """

    if not path.exists():
        return False, "file_not_found"

    if not path.is_file():
        return False, "not_a_regular_file"

    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"stat_failed: {exc}"

    if size < MIN_FILE_SIZE_BYTES:
        return False, "empty_file"

    return True, None


def try_load_nifti(
    path: Path,
) -> Tuple[Optional[nib.spatialimages.SpatialImage], Optional[str]]:
    """
    Load NIfTI without crashing discovery phase.

    Returns:
        image, None

    or:
        None, error_description
    """

    ok, reason = check_file_basic(path)

    if not ok:
        return None, reason

    try:
        img = nib.load(str(path))

        # Force header/shape access.
        _ = img.shape
        _ = img.affine

        return img, None

    except Exception as exc:
        return None, f"unreadable_nifti: {type(exc).__name__}: {exc}"


def load_nifti_strict(
    path: Path,
):
    """
    Strict loader used after preflight filtering.
    """

    ok, reason = check_file_basic(path)

    if not ok:
        raise RuntimeError(
            f"NIfTI file is invalid:\n"
            f"{path}\n"
            f"Reason: {reason}"
        )

    try:
        return nib.load(str(path))

    except Exception as exc:
        raise RuntimeError(
            f"Failed to read NIfTI:\n"
            f"{path}\n"
            f"Error: {exc}"
        ) from exc


# ============================================================
# Geometry validation
# ============================================================

def validate_geometry(
    image_path: Path,
    mask_path: Path,
    case_id: str,
):
    image = load_nifti_strict(image_path)
    mask = load_nifti_strict(mask_path)

    if image.shape != mask.shape:
        raise RuntimeError(
            "\n"
            + "=" * 80
            + "\nATLAS SHAPE MISMATCH\n"
            + "=" * 80
            + "\n"
            f"Case: {case_id}\n\n"
            f"T1:\n{image_path}\n"
            f"Shape: {image.shape}\n\n"
            f"Mask:\n{mask_path}\n"
            f"Shape: {mask.shape}\n\n"
            "This is not treated as an empty/corrupt-file issue.\n"
            "Conversion is intentionally stopped."
        )

    if not np.allclose(
        image.affine,
        mask.affine,
        rtol=1e-5,
        atol=1e-4,
    ):
        raise RuntimeError(
            "\n"
            + "=" * 80
            + "\nATLAS AFFINE MISMATCH\n"
            + "=" * 80
            + "\n"
            f"Case: {case_id}\n\n"
            f"T1:\n{image_path}\n\n"
            f"T1 affine:\n"
            f"{image.affine}\n\n"
            f"Mask:\n{mask_path}\n\n"
            f"Mask affine:\n"
            f"{mask.affine}\n\n"
            "No automatic resampling or registration is performed."
        )


# ============================================================
# Segmentation validation
# ============================================================

def inspect_binary_mask(
    mask_path: Path,
):
    """
    Validate binary mask while tolerating tiny floating-point
    representation errors.

    Accepted:
        0
        1
        0.0000000001
        0.9999999997

    Not accepted:
        0.5
        2
        255
        NaN
        Inf
    """

    img = load_nifti_strict(mask_path)

    try:
        data = np.asanyarray(img.dataobj)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read mask voxel data:\n"
            f"{mask_path}\n"
            f"{exc}"
        ) from exc

    if data.size == 0:
        raise RuntimeError(
            f"Mask contains no voxels:\n"
            f"{mask_path}"
        )

    if not np.all(np.isfinite(data)):
        raise RuntimeError(
            f"Mask contains NaN or Inf:\n"
            f"{mask_path}"
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
                f"... total_unique={len(values)}"
            )
        else:
            value_text = values.tolist()

        raise RuntimeError(
            "\n"
            + "=" * 80
            + "\nINVALID ATLAS SEGMENTATION VALUES\n"
            + "=" * 80
            + "\n"
            f"Mask:\n{mask_path}\n\n"
            f"Values:\n{value_text}\n\n"
            f"Maximum integer deviation:\n"
            f"{max_error}\n\n"
            f"Allowed tolerance:\n"
            f"{LABEL_INTEGER_TOLERANCE}\n"
        )

    rounded_values = np.unique(rounded)

    if not np.all(
        np.isin(
            rounded_values,
            [0, 1],
        )
    ):
        raise RuntimeError(
            "\n"
            + "=" * 80
            + "\nINVALID ATLAS LABEL CLASSES\n"
            + "=" * 80
            + "\n"
            f"Mask:\n{mask_path}\n\n"
            f"Rounded classes:\n"
            f"{rounded_values.tolist()}\n\n"
            "Expected only:\n"
            "[0, 1]"
        )

    lesion_voxels = int(
        np.count_nonzero(
            rounded == 1
        )
    )

    return {
        "max_rounding_error": max_error,
        "values": rounded_values.tolist(),
        "lesion_voxels": lesion_voxels,
    }


# ============================================================
# Save standardized mask
# ============================================================

def save_binary_mask_uint8(
    src: Path,
    dst: Path,
):
    """
    Standardize valid binary segmentation to uint8 {0,1}.

    Spatial geometry is preserved.
    """

    img = load_nifti_strict(src)

    data = np.asanyarray(
        img.dataobj
    )

    binary = np.rint(
        data
    ).astype(
        np.uint8
    )

    values = np.unique(binary)

    if not np.all(
        np.isin(
            values,
            [0, 1],
        )
    ):
        raise RuntimeError(
            f"Unexpected output classes after normalization:\n"
            f"{src}\n"
            f"{values.tolist()}"
        )

    header = img.header.copy()

    header.set_data_dtype(
        np.uint8
    )

    output_img = nib.Nifti1Image(
        binary,
        img.affine,
        header=header,
    )

    # Preserve qform / sform.
    qform, qform_code = img.get_qform(
        coded=True
    )

    sform, sform_code = img.get_sform(
        coded=True
    )

    if qform is not None:
        output_img.set_qform(
            qform,
            int(qform_code),
        )

    if sform is not None:
        output_img.set_sform(
            sform,
            int(sform_code),
        )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        output_img,
        str(dst),
    )


# ============================================================
# MRI copying
# ============================================================

def copy_nifti(
    src: Path,
    dst: Path,
):
    """
    Copy T1 exactly as provided.
    """

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        src,
        dst,
    )


# ============================================================
# Dataset discovery + corrupted file filtering
# ============================================================

def discover_atlas_cases(
    source_root: Path,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Find ATLAS T1-mask pairs.

    Empty/corrupted/missing NIfTI files are excluded
    and recorded instead of crashing conversion.
    """

    if not source_root.exists():
        raise FileNotFoundError(
            f"ATLAS source directory does not exist:\n"
            f"{source_root}"
        )

    t1_files = sorted(
        source_root.glob(
            "R*/sub-*/ses-*/anat/"
            "*_desc-brainnorm_T1w.nii.gz"
        )
    )

    if not t1_files:
        t1_files = sorted(
            source_root.rglob(
                "*_desc-brainnorm_T1w.nii.gz"
            )
        )

    if not t1_files:
        raise RuntimeError(
            "No ATLAS T1 images were found.\n\n"
            f"Source:\n"
            f"{source_root}"
        )

    print(
        f"[INFO] Found {len(t1_files)} candidate ATLAS T1 files."
    )

    valid_cases: List[Dict] = []
    excluded_cases: List[Dict] = []

    used_case_ids = set()

    for index, t1_path in enumerate(
        t1_files,
        start=1,
    ):
        anat_dir = t1_path.parent
        session_dir = anat_dir.parent
        subject_dir = session_dir.parent

        subject = subject_dir.name
        session = session_dir.name

        if not subject.startswith("sub-"):

            excluded_cases.append(
                {
                    "source_t1": str(t1_path),
                    "reason": "invalid_subject_directory_name",
                }
            )

            continue

        if not session.startswith("ses-"):

            excluded_cases.append(
                {
                    "source_t1": str(t1_path),
                    "reason": "invalid_session_directory_name",
                }
            )

            continue

        case_id = build_case_id(
            subject,
            session,
        )

        if case_id in used_case_ids:
            raise RuntimeError(
                f"Duplicate ATLAS case ID detected:\n"
                f"{case_id}"
            )

        used_case_ids.add(case_id)

        # ----------------------------------------------------
        # Find lesion mask
        # ----------------------------------------------------

        mask_candidates = sorted(
            anat_dir.glob(
                "*_label-lesion_desc-T1lesion_mask.nii.gz"
            )
        )

        if len(mask_candidates) == 0:

            excluded_cases.append(
                {
                    "case_id": case_id,
                    "subject": subject,
                    "session": session,
                    "t1": str(t1_path),
                    "mask": None,
                    "reason": "missing_mask",
                }
            )

            continue

        if len(mask_candidates) > 1:
            raise RuntimeError(
                f"\nMultiple lesion masks found for {case_id}:\n"
                + "\n".join(
                    str(x)
                    for x in mask_candidates
                )
            )

        mask_path = mask_candidates[0]

        # ----------------------------------------------------
        # Check T1
        # ----------------------------------------------------

        _, t1_error = try_load_nifti(
            t1_path
        )

        if t1_error is not None:

            excluded_cases.append(
                {
                    "case_id": case_id,
                    "subject": subject,
                    "session": session,
                    "t1": str(t1_path),
                    "mask": str(mask_path),
                    "reason": f"invalid_t1: {t1_error}",
                }
            )

            continue

        # ----------------------------------------------------
        # Check mask
        # ----------------------------------------------------

        _, mask_error = try_load_nifti(
            mask_path
        )

        if mask_error is not None:

            excluded_cases.append(
                {
                    "case_id": case_id,
                    "subject": subject,
                    "session": session,
                    "t1": str(t1_path),
                    "mask": str(mask_path),
                    "reason": f"invalid_mask: {mask_error}",
                }
            )

            continue

        # ----------------------------------------------------
        # Good pair
        # ----------------------------------------------------

        valid_cases.append(
            {
                "case_id": case_id,
                "subject": subject,
                "session": session,
                "t1": t1_path,
                "mask": mask_path,
            }
        )

        if (
            index == 1
            or index % 200 == 0
            or index == len(t1_files)
        ):
            print(
                f"[DISCOVER] "
                f"{index:4d}/{len(t1_files):4d}"
            )

    if not valid_cases:
        raise RuntimeError(
            "No valid ATLAS T1-mask pairs remained "
            "after integrity screening."
        )

    return valid_cases, excluded_cases


# ============================================================
# Full validation of valid cases
# ============================================================

def validate_cases(
    cases: List[Dict],
):
    """
    Validate actual image/mask content after corrupted files
    have been excluded.
    """

    print(
        "\n[INFO] Validating remaining ATLAS dataset..."
    )

    total = len(cases)

    floating_masks = 0
    zero_lesion_masks = 0
    max_rounding_error = 0.0

    for index, case in enumerate(
        cases,
        start=1,
    ):
        case_id = case["case_id"]
        t1_path = Path(case["t1"])
        mask_path = Path(case["mask"])

        # Geometry problems are NOT silently ignored.
        validate_geometry(
            t1_path,
            mask_path,
            case_id,
        )

        info = inspect_binary_mask(
            mask_path
        )

        error = info[
            "max_rounding_error"
        ]

        max_rounding_error = max(
            max_rounding_error,
            error,
        )

        if error > 0:
            floating_masks += 1

        if info["lesion_voxels"] == 0:
            zero_lesion_masks += 1

        if (
            index == 1
            or index % 50 == 0
            or index == total
        ):
            print(
                f"[VALIDATE] "
                f"{index:4d}/{total:4d} "
                f"{case_id}"
            )

    print(
        "[OK] Remaining ATLAS cases passed validation."
    )

    print(
        f"[INFO] Floating-point masks: "
        f"{floating_masks}"
    )

    print(
        f"[INFO] Maximum rounding error: "
        f"{max_rounding_error:.12g}"
    )

    print(
        f"[INFO] Valid masks containing zero lesion voxels: "
        f"{zero_lesion_masks}"
    )

    if zero_lesion_masks > 0:
        print(
            "[WARNING] Some readable masks contain no foreground lesion."
        )


# ============================================================
# Output directory
# ============================================================

def prepare_output_directory(
    target_root: Path,
    clean: bool,
):
    images_tr = (
        target_root
        / "imagesTr"
    )

    labels_tr = (
        target_root
        / "labelsTr"
    )

    images_ts = (
        target_root
        / "imagesTs"
    )

    dataset_json = (
        target_root
        / "dataset.json"
    )

    excluded_json = (
        target_root
        / "excluded_cases.json"
    )

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

    for file_path in [
        dataset_json,
        excluded_json,
    ]:
        if file_path.exists():
            existing.append(
                str(file_path)
            )

    if existing and not clean:
        raise RuntimeError(
            "\nDataset902 already contains generated files:\n\n"
            + "\n".join(existing)
            + "\n\n"
            "Use --clean to rebuild Dataset902."
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

        for file_path in [
            dataset_json,
            excluded_json,
        ]:
            if file_path.exists():
                file_path.unlink()

    images_tr.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_tr.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        images_tr,
        labels_tr,
    )


# ============================================================
# Save exclusions
# ============================================================

def save_excluded_cases(
    target_root: Path,
    excluded_cases: List[Dict],
    total_candidates: int,
    total_valid: int,
):
    output = {
        "dataset": "ATLAS",
        "total_candidates": int(total_candidates),
        "total_valid": int(total_valid),
        "total_excluded": int(
            len(excluded_cases)
        ),
        "excluded_cases": excluded_cases,
    }

    path = (
        target_root
        / "excluded_cases.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"[OK] Exclusion report written:\n"
        f"{path}"
    )


# ============================================================
# dataset.json
# ============================================================

def create_dataset_json(
    target_root: Path,
    num_training: int,
):
    dataset = {
        "name": "Brain_Atlas2",
        "description": (
            "ATLAS stroke lesion segmentation dataset "
            "using brain-normalized T1-weighted MRI"
        ),
        "channel_names": {
            "0": "T1",
        },
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
# Output validation
# ============================================================

def validate_output(
    target_root: Path,
    cases: List[Dict],
):
    images_tr = (
        target_root
        / "imagesTr"
    )

    labels_tr = (
        target_root
        / "labelsTr"
    )

    image_files = sorted(
        images_tr.glob(
            "*_0000.nii.gz"
        )
    )

    label_files = sorted(
        labels_tr.glob(
            "*.nii.gz"
        )
    )

    expected = len(cases)

    if len(image_files) != expected:
        raise RuntimeError(
            f"Output image count mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual: {len(image_files)}"
        )

    if len(label_files) != expected:
        raise RuntimeError(
            f"Output label count mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual: {len(label_files)}"
        )

    print(
        "\n[INFO] Verifying generated Dataset902..."
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):
        case_id = case["case_id"]

        image_path = (
            images_tr
            / f"{case_id}_0000.nii.gz"
        )

        label_path = (
            labels_tr
            / f"{case_id}.nii.gz"
        )

        if not image_path.exists():
            raise RuntimeError(
                f"Missing output T1:\n"
                f"{image_path}"
            )

        if not label_path.exists():
            raise RuntimeError(
                f"Missing output mask:\n"
                f"{label_path}"
            )

        label_img = nib.load(
            str(label_path)
        )

        label_data = np.asanyarray(
            label_img.dataobj
        )

        values = np.unique(
            label_data
        )

        if not np.all(
            np.isin(
                values,
                [0, 1],
            )
        ):
            raise RuntimeError(
                f"Generated mask contains invalid values:\n"
                f"{label_path}\n"
                f"{values.tolist()}"
            )

        dtype = (
            label_img.header.get_data_dtype()
        )

        if dtype != np.dtype(np.uint8):
            raise RuntimeError(
                f"Generated segmentation is not uint8:\n"
                f"{label_path}\n"
                f"dtype={dtype}"
            )

        if (
            index == 1
            or index % 100 == 0
            or index == expected
        ):
            print(
                f"[OUTPUT CHECK] "
                f"{index:4d}/{expected:4d}"
            )

    print(
        "[OK] Dataset902 output validation passed."
    )


# ============================================================
# Main conversion
# ============================================================

def convert(
    source_root: Path,
    target_root: Path,
    clean: bool,
):
    print(
        "=" * 80
    )

    print(
        "ATLAS -> nnU-Net v2 conversion"
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

    # --------------------------------------------------------
    # Step 1: discovery + basic integrity filtering
    # --------------------------------------------------------

    cases, excluded_cases = discover_atlas_cases(
        source_root
    )

    total_candidates = (
        len(cases)
        + len(excluded_cases)
    )

    print()

    print(
        f"[INFO] Candidate cases : "
        f"{total_candidates}"
    )

    print(
        f"[INFO] Valid pairs     : "
        f"{len(cases)}"
    )

    print(
        f"[INFO] Excluded cases  : "
        f"{len(excluded_cases)}"
    )

    if excluded_cases:

        print(
            "\n[WARNING] Invalid/corrupted cases detected:"
        )

        for item in excluded_cases:

            print(
                f"  - {item.get('case_id', 'unknown')}: "
                f"{item.get('reason')}"
            )

    # --------------------------------------------------------
    # Step 2: strict validation of remaining cases
    # --------------------------------------------------------

    validate_cases(
        cases
    )

    # --------------------------------------------------------
    # Step 3: output
    # --------------------------------------------------------

    images_tr, labels_tr = (
        prepare_output_directory(
            target_root,
            clean,
        )
    )

    # Save audit record before image conversion.
    save_excluded_cases(
        target_root=target_root,
        excluded_cases=excluded_cases,
        total_candidates=total_candidates,
        total_valid=len(cases),
    )

    # --------------------------------------------------------
    # Step 4: write files
    # --------------------------------------------------------

    print(
        "\n[INFO] Writing Dataset902..."
    )

    total = len(cases)

    for index, case in enumerate(
        cases,
        start=1,
    ):
        case_id = (
            case["case_id"]
        )

        t1_src = Path(
            case["t1"]
        )

        mask_src = Path(
            case["mask"]
        )

        t1_dst = (
            images_tr
            / f"{case_id}_0000.nii.gz"
        )

        mask_dst = (
            labels_tr
            / f"{case_id}.nii.gz"
        )

        # T1: exact copy
        copy_nifti(
            t1_src,
            t1_dst,
        )

        # Mask: true uint8 binary representation
        save_binary_mask_uint8(
            mask_src,
            mask_dst,
        )

        if (
            index == 1
            or index % 50 == 0
            or index == total
        ):
            print(
                f"[WRITE] "
                f"{index:4d}/{total:4d} "
                f"{case_id}"
            )

    # --------------------------------------------------------
    # Step 5: dataset.json
    # --------------------------------------------------------

    create_dataset_json(
        target_root,
        len(cases),
    )

    # --------------------------------------------------------
    # Step 6: output validation
    # --------------------------------------------------------

    validate_output(
        target_root,
        cases,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "[SUCCESS] ATLAS conversion completed"
    )

    print(
        "=" * 80
    )

    print(
        f"Candidate cases : {total_candidates}"
    )

    print(
        f"Included cases  : {len(cases)}"
    )

    print(
        f"Excluded cases  : {len(excluded_cases)}"
    )

    print(
        f"Images          : {len(cases)}"
    )

    print(
        f"Labels          : {len(cases)}"
    )

    print(
        f"Output          : {target_root}"
    )

    print(
        f"Exclusion log   : "
        f"{target_root / 'excluded_cases.json'}"
    )

    print()

    print(
        "Next command:"
    )

    print(
        "nnUNetv2_plan_and_preprocess "
        "-d 902 --verify_dataset_integrity"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert ATLAS stroke lesion dataset "
            "to nnU-Net v2 raw dataset format."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=(
            f"ATLAS source. "
            f"Default: {DEFAULT_SOURCE_ROOT}"
        ),
    )

    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help=(
            f"nnU-Net target. "
            f"Default: {DEFAULT_TARGET_ROOT}"
        ),
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Remove previously generated Dataset902 "
            "imagesTr/labelsTr/dataset.json/exclusion log."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        convert(
            source_root=args.source.resolve(),
            target_root=args.target.resolve(),
            clean=args.clean,
        )

    except Exception as exc:

        print()

        print(
            "=" * 80
        )

        print(
            "[FAILED] ATLAS conversion failed"
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