#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset802_LungNodel.py

Regenerate Dataset801_LungNodel with corrected Hounsfield units.

WHY THIS EXISTS
---------------
Dataset801's `imagesTr` is NOT raw HU. Measured across 30 cases:

    min == -1350.0 and max == 150.0 for EVERY case   ->  a fixed window was applied
    air pile-up at the floor (0.1%-27% of voxels)    ->  the offset pushes air below it
    max == 150 everywhere                            ->  bone/contrast is already gone

A correctly scaled chest CT has air near -1000 HU and bone well above +500 HU, so air
cannot sit on a -1350 floor. The stored values are a fixed window shifted down:

    stored = clip(HU, -1000, 500) - 350        (inverse:  HU = stored + 350)

Anchoring on air reproduces the whole anatomy consistently: air -1350 -> -1000,
lung parenchyma -1180 -> -830, ground-glass -1035 -> -685, solid -253 -> +97.
That is why --hu-offset defaults to 350.

APPROXIMATE, NOT EXACT
----------------------
Without the original DICOM there is no per-voxel ground truth, so the +350 HU shift
cannot be verified as the exact scanner value for any individual voxel. All that can be
shown is that it reproduces a consistent fixed-window-plus-constant-offset model, using
air and lung tissue as the landmarks.

Everything that was denser than 500 HU was already flattened onto the 150 rail before
this script ran. Adding the offset back maps it to 500 and cannot separate bone,
contrast or dense mediastinum again. Within the retained window the intensity ordering
is preserved; above the rail it is gone. Do not describe the result as "recovered HU"
in a write-up - describe it as an offset-reconstructed approximation with a documented
irreversible loss above 500 HU.

WHAT IS CARRIED OVER FROM THE 801 SCRIPT
----------------------------------------
All of its QC proved its worth - `strict_check_ct_with_sitk` is what rejected the
corrupt NFMKY-123.nii.gz - so the numeric validation, mask binarisation, geometry
handling and the final re-read audit are kept as-is. The differences:

  1. Images are REWRITTEN (float32, offset applied) instead of copied/hard-linked.
  2. A HU plausibility check was added. The 801 script had no such guard, which is
     exactly why the windowing went unnoticed until the fingerprint looked wrong.

OUTPUT
------
Dataset802_LungNodel/
    imagesTr/  float32, corrected HU
    labelsTr/  uint8 {0, 1}, geometry copied from the rewritten image
    dataset.json
    conversion_report.csv
    skipped_cases.csv

USAGE
-----
python nnunetv2/dataset_conversion/Dataset802_LungNodel.py --clean --workers 4

then

nnUNetv2_plan_and_preprocess -d 802 --verify_dataset_integrity
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation, ornt_transform
from nibabel.processing import resample_from_to

# =============================================================================
# Configuration
# =============================================================================

DATASET_ID = 802
DATASET_NAME = "LungNodel"
CASE_PREFIX = "LungNodel"
LABEL_NAME = "lung_nodule"

DEFAULT_SOURCE_ROOT = Path(
    "/data/sdb/medical/datasets/ct_liver/CT/2021-2023第一批已勾画"
)
DEFAULT_TARGET_ROOT = Path(
    "/data/sdb/medical/nnunet_data/nnUNet_raw/Dataset802_LungNodel"
)

IMAGE_DIR_NAME = "images"
MASK_DIR_NAME = "masks"

# HU = stored + offset. See the module docstring for how 350 was derived.
DEFAULT_HU_OFFSET = 350.0

# Landmarks used to sanity-check the corrected volumes. A chest CT must contain air
# (near -1000 HU), and the corrected range must reach the expected upper rail.
#
# The upper-rail test is NOT a bone test. Every case corrects to max == 500 purely
# because 150 + 350 == 500, which says nothing about whether bone is present - all
# true values of 500, 800, 1200, 2000 HU are already merged onto that rail. What the
# test actually verifies is that the offset was applied at all: forget it and the
# range collapses to [-1000, 150], which fails loudly.
HU_AIR_MAX = -900.0
HU_UPPER_RAIL_MIN = 300.0

AFFINE_RTOL = 1e-5
AFFINE_ATOL = 1e-4
SITK_GEOMETRY_RTOL = 1e-5
SITK_GEOMETRY_ATOL = 1e-4
LABEL_INTEGER_TOLERANCE = 1e-3
MIN_FOV_OVERLAP_OF_SMALLER = 0.05

DEFAULT_WORKERS = 4
DEFAULT_PROGRESS_EVERY = 10
FLOAT32_INFO = np.finfo(np.float32)

IDENTITY_ORIENTATION = np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=float)

try:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
except Exception:
    pass


# =============================================================================
# Generic utilities
# =============================================================================

def strip_nii_gz(name: str) -> str:
    if not name.lower().endswith(".nii.gz"):
        raise ValueError(f"Not a .nii.gz file: {name}")
    return name[:-7]


def natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def sanitize_case_name(text: str) -> str:
    text = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9]+", "_", text.strip())).strip("_")
    if not text:
        raise ValueError(f"Cannot build case ID from: {text!r}")
    return text


def build_case_id(filename: str) -> str:
    return f"{CASE_PREFIX}_{sanitize_case_name(strip_nii_gz(filename))}"


def list_nii_gz(directory: Path) -> List[Path]:
    with os.scandir(directory) as it:
        files = [Path(e.path) for e in it
                 if e.is_file() and e.name.lower().endswith(".nii.gz")]
    return sorted(files, key=lambda p: natural_key(p.name))


def load_nifti(path: Path) -> nib.Nifti1Image:
    try:
        return nib.load(str(path))
    except Exception as exc:
        raise RuntimeError(f"Cannot read NIfTI: {path}\n{exc}") from exc


def orientation_string(affine: np.ndarray) -> str:
    return "".join(nib.aff2axcodes(affine))


def voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


def transformed_shape(shape: Tuple[int, ...], transform: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(v) for v in np.asarray(shape, dtype=int)[
        np.argsort(transform[:, 0].astype(int))])


def safe_unlink(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        if path.exists() or path.is_symlink():
            path.unlink()
    except Exception:
        pass


def short_error_message(exc: BaseException) -> str:
    return str(exc).strip() or exc.__class__.__name__


def first_error_line(exc: BaseException) -> str:
    text = short_error_message(exc)
    return text.splitlines()[0] if text else exc.__class__.__name__


# =============================================================================
# HU correction and plausibility - the reason this script exists
# =============================================================================

def correct_hu(array: np.ndarray, offset: float) -> np.ndarray:
    """stored -> HU. Values on the stored rails stay on rails; nothing is invented."""
    return (array.astype(np.float32, copy=False) + np.float32(offset))


def check_hu_plausibility(array: np.ndarray, path: Path) -> Dict[str, Any]:
    """Reject volumes that still do not look like HU.

    This is the guard the Dataset801 job was missing. Had it been there, the windowed
    source data would have failed at conversion time instead of surfacing later as a
    nonsensical fingerprint.
    """
    lo = float(array.min())
    hi = float(array.max())

    if lo > HU_AIR_MAX:
        raise RuntimeError(
            f"Corrected CT has no air-density voxels: {path}\n"
            f"min={lo:.1f}, expected <= {HU_AIR_MAX:.0f} HU.\n"
            f"Either --hu-offset is wrong or this is not a chest CT.")
    if hi < HU_UPPER_RAIL_MIN:
        raise RuntimeError(
            f"Corrected CT does not reach the expected upper intensity rail: {path}\n"
            f"max={hi:.1f}, expected >= {HU_UPPER_RAIL_MIN:.0f} HU.\n"
            f"A source max of 150 corrects to 150 only when --hu-offset was NOT "
            f"applied (150 + 350 = 500 expected).")

    return {'hu_min': lo, 'hu_max': hi, 'hu_range': hi - lo}


def check_source_window_rails(array: np.ndarray) -> Dict[str, Any]:
    """Record how much of the source sits on a hard clip rail. Informational."""
    lo = float(array.min())
    hi = float(array.max())
    sub = array
    return {
        'source_min': lo,
        'source_max': hi,
        'source_rail_low_frac': float((sub <= lo + 1e-3).mean()),
        'source_rail_high_frac': float((sub >= hi - 1e-3).mean()),
    }


# =============================================================================
# Numeric validation (kept from the 801 script)
# =============================================================================

def read_sitk_image(path: Path) -> sitk.Image:
    try:
        return sitk.ReadImage(str(path))
    except Exception as exc:
        raise RuntimeError(f"SimpleITK cannot read image: {path}\n{exc}") from exc


def validate_numeric_array_for_nnunet(array: np.ndarray, path: Path, role: str) -> Dict[str, Any]:
    if array.size == 0:
        raise RuntimeError(f"{role} array is empty: {path}")
    if np.iscomplexobj(array):
        raise RuntimeError(f"{role} is complex-valued and unsupported: {path}")

    with np.errstate(over="ignore", invalid="ignore"):
        if not np.all(np.isfinite(array)):
            raise RuntimeError(
                f"{role} contains non-finite voxel values: {path}\n"
                f"NaN={int(np.count_nonzero(np.isnan(array)))}, "
                f"+Inf={int(np.count_nonzero(np.isposinf(array)))}, "
                f"-Inf={int(np.count_nonzero(np.isneginf(array)))}")

    data_min, data_max = float(np.min(array)), float(np.max(array))
    # float() on the bound: comparing a Python float against a np.float32 scalar makes
    # numpy cast the float to float32, which itself overflows ("overflow encountered in
    # cast") for garbage values like 1e307 and trains the reader to ignore warnings.
    f32_max = float(FLOAT32_INFO.max)
    if data_min < -f32_max or data_max > f32_max:
        raise RuntimeError(
            f"{role} contains values outside float32 range: {path}\n"
            f"min={data_min}, max={data_max}")

    with np.errstate(over="ignore", invalid="ignore"):
        if not np.all(np.isfinite(array.astype(np.float32, copy=False))):
            raise RuntimeError(f"{role} becomes NaN/Inf after float32 conversion: {path}")

    return {'min': data_min, 'max': data_max,
            'mean': float(np.mean(array, dtype=np.float64)),
            'dtype': str(array.dtype)}


def strict_check_ct_with_sitk(path: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    image = read_sitk_image(path)
    if image.GetDimension() != 3:
        raise RuntimeError(f"Expected 3D CT: {path} (dim={image.GetDimension()})")
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise RuntimeError(
            f"Expected scalar CT, got {image.GetNumberOfComponentsPerPixel()} "
            f"components: {path}")

    spacing = tuple(float(v) for v in image.GetSpacing())
    if any(v <= 0 or not np.isfinite(v) for v in spacing):
        raise RuntimeError(f"Invalid CT spacing: {path}\nspacing={spacing}")
    if not np.all(np.isfinite(np.asarray(image.GetOrigin(), dtype=np.float64))):
        raise RuntimeError(f"Invalid CT origin: {path}")
    if not np.all(np.isfinite(np.asarray(image.GetDirection(), dtype=np.float64))):
        raise RuntimeError(f"Invalid CT direction: {path}")

    try:
        array = sitk.GetArrayFromImage(image)
    except Exception as exc:
        raise RuntimeError(f"Cannot decode CT voxels with SimpleITK: {path}\n{exc}") from exc

    stats = validate_numeric_array_for_nnunet(array, path, "CT")
    info = {
        **stats,
        'sitk_size': tuple(int(v) for v in image.GetSize()),
        'sitk_spacing': spacing,
        'sitk_origin': tuple(float(v) for v in image.GetOrigin()),
        'sitk_direction': tuple(float(v) for v in image.GetDirection()),
        'sitk_pixel_type': image.GetPixelIDTypeAsString(),
    }
    # Return the decoded array alongside the stats: decompression dominates the runtime,
    # so callers must not pay for it a second time just to reach the voxels.
    return info, array


def compare_sitk_geometry(image: sitk.Image, mask: sitk.Image) -> None:
    for name, a, b in (
        ("size", image.GetSize(), mask.GetSize()),
        ("spacing", image.GetSpacing(), mask.GetSpacing()),
        ("origin", image.GetOrigin(), mask.GetOrigin()),
        ("direction", image.GetDirection(), mask.GetDirection()),
    ):
        if not np.allclose(np.asarray(a, dtype=np.float64),
                           np.asarray(b, dtype=np.float64),
                           rtol=SITK_GEOMETRY_RTOL, atol=SITK_GEOMETRY_ATOL):
            raise RuntimeError(f"SimpleITK image/mask {name} mismatch: {a} != {b}")


def final_audit_written_pair(case_id: str, image_path: Path, mask_path: Path,
                             allow_empty_mask: bool) -> Dict[str, Any]:
    image = read_sitk_image(image_path)
    mask = read_sitk_image(mask_path)

    if image.GetDimension() != 3:
        raise RuntimeError(f"Final CT is not 3D: {image_path}")
    if mask.GetDimension() != 3:
        raise RuntimeError(f"Final mask is not 3D: {mask_path}")
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise RuntimeError(f"Final CT is not scalar: {image_path}")
    if mask.GetNumberOfComponentsPerPixel() != 1:
        raise RuntimeError(f"Final mask is not scalar: {mask_path}")

    compare_sitk_geometry(image, mask)

    image_stats = validate_numeric_array_for_nnunet(
        sitk.GetArrayFromImage(image), image_path, "Final CT")

    mask_array = sitk.GetArrayFromImage(mask)
    if mask_array.size == 0:
        raise RuntimeError(f"Final mask array is empty: {mask_path}")
    if not np.all(np.isfinite(mask_array)):
        raise RuntimeError(f"Final mask contains NaN/Inf: {mask_path}")

    unique = np.unique(mask_array)
    if not np.all(np.isin(unique, [0, 1])):
        raise RuntimeError(
            f"Final mask is not binary: {mask_path}\nvalues={unique.tolist()}")

    foreground = int(np.count_nonzero(mask_array))
    if foreground <= 0 and not allow_empty_mask:
        raise RuntimeError(f"Final mask is empty: {mask_path}")

    return {'case_id': case_id, 'final_image_min': image_stats['min'],
            'final_image_max': image_stats['max'],
            'final_image_mean': image_stats['mean'],
            'final_foreground_voxels': foreground}


# =============================================================================
# Physical FOV helpers
# =============================================================================

def world_aabb(shape: Tuple[int, ...], affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(shape) != 3:
        raise ValueError(f"Expected 3D shape, got {shape}")
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    corners = np.asarray(
        [[0, 0, 0, 1], [nx - 1, 0, 0, 1], [0, ny - 1, 0, 1], [0, 0, nz - 1, 1],
         [nx - 1, ny - 1, 0, 1], [nx - 1, 0, nz - 1, 1], [0, ny - 1, nz - 1, 1],
         [nx - 1, ny - 1, nz - 1, 1]], dtype=np.float64)
    world = (affine @ corners.T).T[:, :3]
    return np.min(world, axis=0), np.max(world, axis=0)


def aabb_overlap_fraction_of_smaller(shape_a, affine_a, shape_b, affine_b) -> float:
    a_min, a_max = world_aabb(shape_a, affine_a)
    b_min, b_max = world_aabb(shape_b, affine_b)
    inter = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    vol_a = float(np.prod(np.maximum(0.0, a_max - a_min)))
    vol_b = float(np.prod(np.maximum(0.0, b_max - b_min)))
    smaller = min(vol_a, vol_b)
    return float(np.prod(inter)) / smaller if smaller > 0 else 0.0


# =============================================================================
# Discovery
# =============================================================================

def make_skip_record(*, case_id, source_name, image_path, mask_path, error_stage,
                     error_type, error_message, traceback_text="") -> Dict[str, Any]:
    return {'case_id': case_id, 'source_name': source_name, 'image_path': image_path,
            'mask_path': mask_path, 'error_stage': error_stage, 'error_type': error_type,
            'error_message': error_message, 'traceback': traceback_text}


def discover_cases(source_root: Path):
    image_root, mask_root = source_root / IMAGE_DIR_NAME, source_root / MASK_DIR_NAME
    for label, d in (("source root", source_root), ("image", image_root),
                     ("mask", mask_root)):
        if not d.is_dir():
            raise FileNotFoundError(f"{label} directory does not exist:\n{d}")

    image_files = list_nii_gz(image_root)
    mask_files = list_nii_gz(mask_root)
    print("\n" + "=" * 80 + "\nSOURCE DISCOVERY\n" + "=" * 80)
    print(f"[INFO] Images : {len(image_files)}")
    print(f"[INFO] Masks  : {len(mask_files)}")
    if not image_files:
        raise RuntimeError(f"No .nii.gz images found in {image_root}")

    mask_by_name = {p.name: p for p in mask_files}
    cases, skipped, used_ids = [], [], set()

    for index, image_path in enumerate(image_files):
        source_name = image_path.name
        try:
            case_id = build_case_id(source_name)
        except Exception as exc:
            skipped.append(make_skip_record(
                case_id="", source_name=source_name, image_path=str(image_path),
                mask_path="", error_stage="discovery", error_type=exc.__class__.__name__,
                error_message=short_error_message(exc)))
            continue

        mask_path = mask_by_name.get(source_name)
        if mask_path is None:
            skipped.append(make_skip_record(
                case_id=case_id, source_name=source_name, image_path=str(image_path),
                mask_path="", error_stage="discovery", error_type="MissingMask",
                error_message="No same-name mask file was found."))
            print(f"[SKIP] {case_id}: missing mask")
            continue

        if case_id in used_ids:
            skipped.append(make_skip_record(
                case_id=case_id, source_name=source_name, image_path=str(image_path),
                mask_path=str(mask_path), error_stage="discovery",
                error_type="DuplicateCaseID",
                error_message=f"Duplicate generated case ID: {case_id}"))
            print(f"[SKIP] {case_id}: duplicate case ID")
            continue

        used_ids.add(case_id)
        cases.append({'index': index, 'source_name': source_name, 'case_id': case_id,
                      'image': str(image_path), 'mask': str(mask_path)})

    image_names = {p.name for p in image_files}
    for mask_path in (p for p in mask_files if p.name not in image_names):
        try:
            case_id = build_case_id(mask_path.name)
        except Exception:
            case_id = ""
        skipped.append(make_skip_record(
            case_id=case_id, source_name=mask_path.name, image_path="",
            mask_path=str(mask_path), error_stage="discovery",
            error_type="OrphanMask",
            error_message="Mask exists but no same-name image was found."))
        print(f"[SKIP] {case_id or mask_path.name}: orphan mask")

    print(f"[OK] Candidate image-mask pairs: {len(cases)}")
    if skipped:
        print(f"[WARNING] Discovery skipped records: {len(skipped)}")
    if not cases:
        raise RuntimeError("No usable image-mask pairs remain after discovery.")
    return cases, skipped


# =============================================================================
# Image / mask inspection
# =============================================================================

def inspect_image_header(image: nib.Nifti1Image, image_path: Path) -> Dict[str, Any]:
    if image.ndim != 3:
        raise RuntimeError(f"Expected 3D CT.\nFile: {image_path}\nShape: {image.shape}")
    zooms = tuple(float(v) for v in image.header.get_zooms()[:3])
    if any((not np.isfinite(v)) or v <= 0 for v in zooms):
        raise RuntimeError(f"Invalid CT spacing.\nFile: {image_path}\nSpacing: {zooms}")
    if not np.all(np.isfinite(image.affine)):
        raise RuntimeError(f"CT affine contains NaN/Inf:\n{image_path}")
    return {'shape': tuple(int(v) for v in image.shape), 'spacing': zooms,
            'orientation': orientation_string(image.affine),
            'dtype': str(image.header.get_data_dtype())}


def inspect_and_binarize_mask(mask: nib.Nifti1Image, mask_path: Path,
                              allow_empty_mask: bool):
    if mask.ndim != 3:
        raise RuntimeError(f"Expected 3D mask.\nFile: {mask_path}\nShape: {mask.shape}")
    if not np.all(np.isfinite(mask.affine)):
        raise RuntimeError(f"Mask affine contains NaN/Inf:\n{mask_path}")

    data = np.asanyarray(mask.dataobj)
    if data.size == 0:
        raise RuntimeError(f"Empty mask volume: {mask_path}")
    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"Mask contains NaN/Inf: {mask_path}")

    rounded = np.rint(data)
    max_error = float(np.max(np.abs(data.astype(np.float64, copy=False)
                                  - rounded.astype(np.float64, copy=False))))
    if max_error > LABEL_INTEGER_TOLERANCE:
        raise RuntimeError(
            "Mask contains non-integer values.\n"
            f"File: {mask_path}\n"
            f"Values preview: {np.unique(data)[:30].tolist()}\n"
            f"Max integer error: {max_error}")
    if np.any(rounded < 0):
        raise RuntimeError(f"Mask contains negative labels:\n{mask_path}\n"
                           f"Values: {np.unique(rounded).tolist()}")

    original_values = np.unique(rounded).tolist()
    binary = (rounded > 0).astype(np.uint8, copy=False)
    foreground = int(np.count_nonzero(binary))
    if foreground <= 0 and not allow_empty_mask:
        raise RuntimeError(f"Mask is empty: {mask_path}")

    return binary, {'shape': tuple(int(v) for v in mask.shape),
                    'orientation': orientation_string(mask.affine),
                    'original_values': original_values,
                    'foreground_voxels': foreground,
                    'max_rounding_error': max_error}


# =============================================================================
# Geometry (unchanged from the 801 script)
# =============================================================================

def analyze_geometry(image: nib.Nifti1Image, mask: nib.Nifti1Image,
                     allow_resample: bool) -> Dict[str, Any]:
    image_shape = tuple(int(v) for v in image.shape)
    mask_shape = tuple(int(v) for v in mask.shape)
    same_shape = image_shape == mask_shape
    same_affine = np.allclose(image.affine, mask.affine, rtol=AFFINE_RTOL,
                              atol=AFFINE_ATOL)
    same_orientation = (nib.aff2axcodes(image.affine) == nib.aff2axcodes(mask.affine))

    if same_shape and same_affine:
        return {'action': 'identity', 'transform': IDENTITY_ORIENTATION,
                'same_shape': True, 'same_affine': True,
                'same_orientation': same_orientation, 'fov_overlap': 1.0}

    image_ornt, mask_ornt = io_orientation(image.affine), io_orientation(mask.affine)
    if not np.isnan(image_ornt).any() and not np.isnan(mask_ornt).any():
        transform = ornt_transform(mask_ornt, image_ornt)
        output_shape = transformed_shape(mask_shape, transform)
        affine_after_ornt = mask.affine @ inv_ornt_aff(transform, mask.shape)
        if output_shape == image_shape and np.allclose(
                affine_after_ornt, image.affine, rtol=AFFINE_RTOL, atol=AFFINE_ATOL):
            return {'action': 'reorient', 'transform': transform,
                    'same_shape': same_shape, 'same_affine': same_affine,
                    'same_orientation': same_orientation, 'fov_overlap': 1.0}

    overlap = aabb_overlap_fraction_of_smaller(image_shape, image.affine,
                                               mask_shape, mask.affine)
    if not allow_resample:
        raise RuntimeError(
            "IMAGE/MASK GEOMETRY MISMATCH\n"
            f"Image shape: {image.shape}\n"
            f"Mask shape : {mask.shape}\n"
            f"Image orientation: {nib.aff2axcodes(image.affine)}\n"
            f"Mask orientation : {nib.aff2axcodes(mask.affine)}\n"
            f"Physical FOV overlap: {overlap:.6f}\n"
            "This case is skipped by default.")
    if not np.isfinite(overlap) or overlap < MIN_FOV_OVERLAP_OF_SMALLER:
        raise RuntimeError(f"Unsafe mask resampling rejected.\n"
                           f"FOV overlap fraction: {overlap:.6f}")

    return {'action': 'resample', 'transform': None, 'same_shape': same_shape,
            'same_affine': same_affine, 'same_orientation': same_orientation,
            'fov_overlap': float(overlap)}


def convert_mask_array(image: nib.Nifti1Image, mask: nib.Nifti1Image,
                       binary: np.ndarray, geometry: Dict[str, Any]) -> np.ndarray:
    action = geometry['action']
    if action == 'identity':
        output = binary
    elif action == 'reorient':
        output = apply_orientation(binary, geometry['transform']).astype(np.uint8,
                                                                        copy=False)
    elif action == 'resample':
        source_seg = nib.Nifti1Image(binary, mask.affine)
        resampled = resample_from_to(
            source_seg, (tuple(int(v) for v in image.shape), image.affine),
            order=0, mode='constant', cval=0.0)
        output = (np.asanyarray(resampled.dataobj) > 0).astype(np.uint8, copy=False)
    else:
        raise RuntimeError(f"Unsupported geometry action: {action}")

    if tuple(output.shape) != tuple(image.shape):
        raise RuntimeError(f"Converted mask shape mismatch: "
                           f"{output.shape} != {image.shape}")
    return output


# =============================================================================
# Output writing
# =============================================================================

def write_corrected_image(stored: np.ndarray, sitk_info: Dict[str, Any], dst: Path,
                          offset: float) -> None:
    """Add the HU offset and write float32 with the source geometry.

    Unlike the Dataset801 job this CANNOT be a copy or hard-link - the voxel values
    change. float32 halves the source's float64 footprint and is what nnU-Net casts to
    anyway.

    Geometry comes from `sitk_info` rather than a live SimpleITK image: fetching it that
    way would mean holding a second full copy of the volume per worker (a 512x512x336
    float64 CT is ~700 MB), and decompression already dominates the runtime.
    """
    out = sitk.GetImageFromArray(correct_hu(stored, offset))
    out.SetSpacing(sitk_info['sitk_spacing'])
    out.SetOrigin(sitk_info['sitk_origin'])
    out.SetDirection(sitk_info['sitk_direction'])
    sitk.WriteImage(out, str(dst), useCompression=True)


def write_mask_like_image(output_nib_order: np.ndarray, image: sitk.Image,
                          dst: Path) -> None:
    """Write the mask on the rewritten image's grid.

    AXIS ORDER IS LOAD-BEARING. `output_nib_order` comes from nibabel, which indexes
    (x, y, z); SimpleITK indexes (z, y, x) and `GetImageFromArray` treats the FIRST
    numpy axis as z. Handing a nibabel-ordered array straight to GetImageFromArray
    transposes the mask relative to the image, and CopyInformation would then raise a
    size mismatch - or, if the volume happens to be cubic, produce a silently rotated
    mask. The axes are reversed explicitly and the grid is asserted below.
    """
    arr_zyx = np.ascontiguousarray(np.transpose(output_nib_order, (2, 1, 0)))
    mask = sitk.GetImageFromArray(arr_zyx.astype(np.uint8, copy=False))

    if tuple(mask.GetSize()) != tuple(image.GetSize()):
        raise RuntimeError(
            "mask/image grid mismatch after axis reversal: "
            f"{mask.GetSize()} vs {image.GetSize()}")

    mask.CopyInformation(image)
    sitk.WriteImage(mask, str(dst), useCompression=True)


# =============================================================================
# One-case worker
# =============================================================================

def process_case(case: Dict[str, Any], output_root: Optional[str],
                 validate_only: bool, allow_empty_mask: bool, allow_resample: bool,
                 hu_offset: float) -> Dict[str, Any]:
    start = time.perf_counter()
    case_id = str(case['case_id'])
    image_path, mask_path = Path(case['image']), Path(case['mask'])

    image = load_nifti(image_path)
    image_info = inspect_image_header(image, image_path)

    # Mandatory full decode - this is what rejected the corrupt NFMKY-123. The decoded
    # array is kept and reused; decompression is the dominant cost per case.
    sitk_info, stored_array = strict_check_ct_with_sitk(image_path)

    mask = load_nifti(mask_path)
    binary, mask_info = inspect_and_binarize_mask(mask, mask_path,
                                                  allow_empty_mask=allow_empty_mask)

    geometry = analyze_geometry(image, mask, allow_resample=allow_resample)
    output_mask = convert_mask_array(image, mask, binary, geometry)

    source_fg = int(np.count_nonzero(binary))
    output_fg = int(np.count_nonzero(output_mask))
    if output_fg <= 0 and not allow_empty_mask:
        raise RuntimeError("Converted mask became empty")
    if geometry['action'] in {'identity', 'reorient'} and source_fg != output_fg:
        raise RuntimeError("Lossless geometry action changed foreground voxel count: "
                           f"{source_fg} -> {output_fg}")

    src_ml = source_fg * voxel_volume_mm3(mask.affine) / 1000.0
    out_ml = output_fg * voxel_volume_mm3(image.affine) / 1000.0

    result: Dict[str, Any] = {
        '_index': int(case['index']), 'case_id': case_id,
        'source_name': str(case['source_name']), 'image_path': str(image_path),
        'mask_path': str(mask_path),
        'image_shape': "x".join(str(v) for v in image_info['shape']),
        'image_spacing': "x".join(f"{v:.8g}" for v in image_info['spacing']),
        'image_orientation': image_info['orientation'],
        'mask_orientation': mask_info['orientation'],
        'same_shape': geometry['same_shape'], 'same_affine': geometry['same_affine'],
        'same_orientation': geometry['same_orientation'],
        'geometry_action': geometry['action'], 'fov_overlap': geometry['fov_overlap'],
        'image_dtype': image_info['dtype'], 'sitk_pixel_type': sitk_info['sitk_pixel_type'],
        'stored_min': sitk_info['min'], 'stored_max': sitk_info['max'],
        'stored_mean': sitk_info['mean'],
        'mask_values_original': json.dumps(mask_info['original_values'],
                                           ensure_ascii=False),
        'source_foreground_voxels': source_fg,
        'output_foreground_voxels': output_fg,
        'source_lesion_volume_ml': src_ml, 'output_lesion_volume_ml': out_ml,
        'volume_ratio': (out_ml / src_ml) if src_ml > 0 else float('nan'),
        'hu_offset_applied': hu_offset,
        'image_write_mode': 'none' if validate_only else 'rewritten_float32',
    }

    if validate_only:
        result.update(check_source_window_rails(stored_array))
        result.update(check_hu_plausibility(correct_hu(stored_array, hu_offset),
                                            image_path))
        result['elapsed_seconds'] = time.perf_counter() - start
        return result

    if output_root is None:
        raise RuntimeError("output_root is required in conversion mode")

    root = Path(output_root)
    image_dst = root / "imagesTr" / f"{case_id}_0000.nii.gz"
    mask_dst = root / "labelsTr" / f"{case_id}.nii.gz"

    try:
        write_corrected_image(stored_array, sitk_info, image_dst, hu_offset)
        result['image_write_mode'] = 'rewritten_float32'
        result['hu_offset_applied'] = hu_offset
        rewritten = read_sitk_image(image_dst)

        # Plausibility is checked on what was actually written, not on the in-memory
        # array, so a write/read round-trip problem cannot slip through.
        corrected_written = sitk.GetArrayFromImage(rewritten)
        result.update(check_source_window_rails(corrected_written - hu_offset))
        result.update(check_hu_plausibility(corrected_written, image_dst))

        write_mask_like_image(output_mask, rewritten, mask_dst)
    except Exception:
        safe_unlink(mask_dst)
        safe_unlink(image_dst)
        raise

    result['elapsed_seconds'] = time.perf_counter() - start
    return result


# =============================================================================
# Parallel conversion
# =============================================================================

def runtime_skip_record(case: Dict[str, Any], exc: BaseException, tb: str,
                        stage: str) -> Dict[str, Any]:
    return make_skip_record(
        case_id=str(case.get('case_id', '')), source_name=str(case.get('source_name', '')),
        image_path=str(case.get('image', '')), mask_path=str(case.get('mask', '')),
        error_stage=stage, error_type=exc.__class__.__name__,
        error_message=short_error_message(exc), traceback_text=tb)


def run_parallel_conversion(cases, output_root, validate_only, allow_empty_mask,
                            allow_resample, hu_offset, workers, progress_every):
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    total = len(cases)
    results, errors = [], []
    stage = "VALIDATE" if validate_only else "CONVERT"

    print("\n" + "=" * 80 + f"\nPARALLEL {stage}\n" + "=" * 80)
    print(f"[INFO] workers          : {workers}")
    print(f"[INFO] HU offset        : {hu_offset:+.1f}  (HU = stored + offset)")
    print(f"[INFO] CT voxel QC      : MANDATORY SimpleITK full decode + HU plausibility")
    print(f"[INFO] allow resample   : {allow_resample}")
    print(f"[INFO] allow empty mask : {allow_empty_mask}")
    print("[INFO] bad-case policy  : SKIP + REPORT + CONTINUE")

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="dataset802_convert") as executor:
        future_to_case = {
            executor.submit(process_case, case,
                            str(output_root) if output_root is not None else None,
                            validate_only, allow_empty_mask, allow_resample,
                            hu_offset): case
            for case in cases}

        completed = 0
        for future in as_completed(future_to_case):
            case = future_to_case[future]
            completed += 1
            try:
                row = future.result()
                results.append(row)
                if (completed == 1 or completed % progress_every == 0
                        or completed == total):
                    elapsed = time.perf_counter() - start
                    speed = completed / elapsed if elapsed > 0 else 0.0
                    print(f"[{stage}] {completed:4d}/{total:4d} ok={len(results):4d} "
                          f"skip={len(errors):3d} last={row['case_id']} "
                          f"HU=[{row.get('hu_min', float('nan')):.0f},"
                          f"{row.get('hu_max', float('nan')):.0f}] "
                          f"speed={speed:.2f} case/s")
            except Exception as exc:
                errors.append(runtime_skip_record(
                    case, exc, traceback.format_exc(),
                    "source_validation" if validate_only else "conversion"))
                print(f"[SKIP] {case['case_id']} ({completed}/{total}): "
                      f"{exc.__class__.__name__}: {first_error_line(exc)}")

    elapsed = time.perf_counter() - start
    results.sort(key=lambda row: row['_index'])
    print(f"\n[OK] {stage} finished: candidates={total}, success={len(results)}, "
          f"skipped={len(errors)}, time={elapsed:.2f}s, "
          f"speed={(total / elapsed if elapsed > 0 else 0.0):.2f} case/s")
    return results, errors


# =============================================================================
# Final staging audit
# =============================================================================

def run_final_staging_audit(rows, staging: Path, allow_empty_mask, workers,
                            progress_every):
    print("\n" + "=" * 80 + "\nFINAL SIMPLEITK / NNUNET COMPATIBILITY AUDIT\n" + "=" * 80)
    if not rows:
        raise RuntimeError("No successful rows available for final audit.")

    row_by_case = {str(r['case_id']): r for r in rows}
    passed, skipped = set(), []
    start = time.perf_counter()
    total = len(rows)

    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="dataset802_audit") as executor:
        future_to_row = {
            executor.submit(final_audit_written_pair, str(row['case_id']),
                            staging / "imagesTr" / f"{row['case_id']}_0000.nii.gz",
                            staging / "labelsTr" / f"{row['case_id']}.nii.gz",
                            allow_empty_mask): row
            for row in rows}

        completed = 0
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            completed += 1
            case_id = str(row['case_id'])
            try:
                row.update(future.result())
                passed.add(case_id)
                if (completed == 1 or completed % progress_every == 0
                        or completed == total):
                    elapsed = time.perf_counter() - start
                    print(f"[FINAL AUDIT] {completed:4d}/{total:4d} "
                          f"pass={len(passed):4d} skip={len(skipped):3d} last={case_id} "
                          f"speed={(completed / elapsed if elapsed > 0 else 0.0):.2f} case/s")
            except Exception as exc:
                image_path = staging / "imagesTr" / f"{case_id}_0000.nii.gz"
                mask_path = staging / "labelsTr" / f"{case_id}.nii.gz"
                safe_unlink(image_path)
                safe_unlink(mask_path)
                skipped.append(make_skip_record(
                    case_id=case_id, source_name=str(row.get('source_name', '')),
                    image_path=str(row.get('image_path', '')),
                    mask_path=str(row.get('mask_path', '')),
                    error_stage='final_output_audit',
                    error_type=exc.__class__.__name__,
                    error_message=short_error_message(exc),
                    traceback_text=traceback.format_exc()))
                print(f"[FINAL SKIP] {case_id}: {first_error_line(exc)}")

    final_rows = [row_by_case[c] for c in row_by_case if c in passed]
    final_rows.sort(key=lambda r: r['_index'])
    return final_rows, skipped


# =============================================================================
# Reporting
# =============================================================================

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def print_summary(final_rows, all_skipped, target_root: Path):
    print("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    print(f"[OK] Final cases written : {len(final_rows)}")
    print(f"[SKIP] Total skipped     : {len(all_skipped)}")

    if not final_rows:
        print("[ERROR] No cases survived.")
        return

    hu_lo = np.array([r['hu_min'] for r in final_rows])
    hu_hi = np.array([r['hu_max'] for r in final_rows])
    print(f"[OK] corrected HU min    : median={np.median(hu_lo):.0f} "
          f"range=[{hu_lo.min():.0f}, {hu_lo.max():.0f}]")
    print(f"[OK] corrected HU max    : median={np.median(hu_hi):.0f} "
          f"range=[{hu_hi.min():.0f}, {hu_hi.max():.0f}]")
    print()
    print("Reminder: everything that was denser than 500 HU in the original scan is")
    print("flattened onto the 500 rail and cannot be recovered. Nodule-range HU")
    print("(-1000..+300) is intact; bone/contrast/mediastinal contrast is not.")


def build_dataset_json(file_ending: str) -> Dict[str, Any]:
    return {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, LABEL_NAME: 1},
        "numTraining": 0,
        "file_ending": file_ending,
    }


def write_dataset_json(target_root: Path, num_training: int, file_ending: str) -> None:
    payload = build_dataset_json(file_ending)
    payload["numTraining"] = int(num_training)
    target_root.mkdir(parents=True, exist_ok=True)
    with open(target_root / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rebuild Dataset801 as Dataset802 with corrected Hounsfield units.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    p.add_argument("--hu-offset", type=float, default=DEFAULT_HU_OFFSET,
                   help="added to every stored voxel to recover HU "
                        "(350 anchors air at -1000)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    p.add_argument("--file-ending", default=".nii.gz")
    p.add_argument("--allow-empty-mask", action="store_true")
    p.add_argument("--allow-resample", action="store_true")
    p.add_argument("--clean", action="store_true",
                   help="delete an existing target dataset first")
    p.add_argument("--validate-only", action="store_true",
                   help="QC only, write nothing")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    if args.clean and not args.validate_only:
        if args.target_root.exists():
            print(f"[INFO] Removing existing target: {args.target_root}")
            shutil.rmtree(args.target_root)
    if not args.validate_only:
        (args.target_root / "imagesTr").mkdir(parents=True, exist_ok=True)
        (args.target_root / "labelsTr").mkdir(parents=True, exist_ok=True)

    cases, discovery_skips = discover_cases(args.source_root)

    results, conversion_skips = run_parallel_conversion(
        cases, None if args.validate_only else args.target_root, args.validate_only,
        args.allow_empty_mask, args.allow_resample, args.hu_offset, args.workers,
        args.progress_every)

    if args.validate_only:
        write_csv(args.target_root / "conversion_report.csv", results)
        write_csv(args.target_root / "skipped_cases.csv",
                  discovery_skips + conversion_skips)
        print_summary(results, discovery_skips + conversion_skips, args.target_root)
        return 0

    final_rows, audit_skips = run_final_staging_audit(
        results, args.target_root, args.allow_empty_mask, args.workers,
        args.progress_every)

    all_skipped = discovery_skips + conversion_skips + audit_skips
    write_csv(args.target_root / "conversion_report.csv", final_rows)
    write_csv(args.target_root / "skipped_cases.csv", all_skipped)
    write_dataset_json(args.target_root, len(final_rows), args.file_ending)
    print_summary(final_rows, all_skipped, args.target_root)

    print(f"\n[OK] dataset.json written with numTraining={len(final_rows)}")
    print("\nNext:")
    print("  nnUNetv2_plan_and_preprocess -d 802 --verify_dataset_integrity")
    print("  rm -rf $nnUNet_preprocessed/Dataset802_LungNodel   # only if re-running")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
