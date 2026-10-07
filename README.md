# Breast-DCE-MRI-Dual-Segments

**Structure-Preserving Cross-Phase Alignment and Uncertainty-Guided Fusion for 3D Breast Tumor Segmentation in DCE-MRI**

> Official repository for **Breast-DCE-MRI-Dual-Segments**, a dual-phase 3D DCE-MRI breast tumor segmentation framework designed to reduce residual cross-phase misalignment and adaptively fuse phase information according to local prediction reliability.

---

## Overview

Dynamic contrast-enhanced magnetic resonance imaging (DCE-MRI) provides complementary information across enhancement phases. Residual spatial misalignment may remain after rigid registration because of respiration, patient motion, and breast deformation. The reliability of the early and delayed enhancement phases may also vary across tumor interiors, boundaries, and low-contrast regions.

Breast-DCE-MRI-Dual-Segments follows an **alignment-before-fusion** strategy:

1. **Phase-specific shallow stems** capture phase-dependent appearance cues.
2. A **shared 3D encoder** maps both enhancement phases into a comparable semantic space.
3. **Structure-Preserving Cross-Phase Alignment (SPCA)** estimates bounded residual displacement fields in a shared structural representation.
4. **Uncertainty-Guided Cross-Phase Fusion (UGCF)** uses supervised predictive entropy, cross-phase prediction disagreement, and feature differences to guide adaptive feature integration.
5. A hierarchical **3D decoder** reconstructs the final volumetric tumor segmentation.

<p align="center">
  <img src="docs/figures/network_architecture.png" width="900" alt="VCAU-Net architecture">
</p>

<p align="center"><b>Figure 1.</b> Overall architecture of VCAU-Net.</p>

---

## Method

### Phase-specific representation and shared encoding

The early and delayed enhancement phases are processed by independent shallow 3D stems. Deeper encoder stages share parameters across phases, providing a common semantic basis for subsequent cross-phase interaction while retaining phase-dependent responses.

### Structure-Preserving Cross-Phase Alignment (SPCA)

SPCA performs residual feature alignment after rigid preprocessing. Rather than directly enforcing similarity between phase-dependent image intensities, the module estimates local correspondence in a shared structural feature space.

Main components include:

- shared structural projection;
- local symmetric cross-phase correspondence;
- bounded bidirectional residual displacement prediction;
- differentiable 3D warping;
- structural alignment, displacement smoothness, and inverse-consistency regularization.

<p align="center">
  <img src="docs/figures/spca.png" width="850" alt="SPCA module">
</p>

### Uncertainty-Guided Cross-Phase Fusion (UGCF)

UGCF uses lightweight auxiliary segmentation heads to obtain phase-specific probability maps. Normalized predictive entropy is used as a local uncertainty proxy, while cross-phase prediction disagreement provides an additional reliability cue.

The fusion pathway combines:

- relative phase reliability;
- joint reliability gating;
- aligned contextual features;
- explicit cross-phase feature differences;
- residual feature fusion.

<p align="center">
  <img src="docs/figures/ugcf.png" width="850" alt="UGCF module">
</p>

---

## Study design

Two DCE-MRI cohorts were used.

### Internal cohort

- **296 patients**
- Guangxi Medical University Cancer Hospital
- Retrospective cohort
- Patient-level **five-fold cross-validation**
- Dual-phase 3D DCE-MRI
- Manual 3D tumor annotations as the reference standard

### Independent external cohort

- **100 public cases**
- Yunnan Cancer Hospital DCE-MRI dataset
- Independent external evaluation
- One pre-contrast and five post-contrast acquisitions are available
- The second and fifth post-contrast acquisitions were used as the early and delayed enhancement inputs
- Phase correspondence was based on relative acquisition order rather than identical post-injection timing between centers

No external case was used for model training, hyperparameter selection, early stopping, or threshold tuning.

---

## Main results

### Internal five-fold cross-validation

| Method | Dice (%) | IoU (%) | Sensitivity (%) | Precision (%) | HD95 (mm) | ASSD (mm) |
|---|---:|---:|---:|---:|---:|---:|
| 3D U-Net | 78.62 ± 14.83 | 67.09 ± 18.94 | 77.31 ± 16.42 | 80.16 ± 15.23 | 23.47 ± 24.86 | 13.96 ± 9.34 |
| V-Net | 79.45 ± 14.21 | 68.09 ± 18.38 | 78.23 ± 15.66 | 80.86 ± 14.91 | 22.16 ± 23.31 | 10.52 ± 8.88 |
| nnU-Net v2 | 84.96 ± 11.84 | 75.53 ± 16.52 | 83.74 ± 13.12 | 86.39 ± 12.26 | 16.82 ± 18.37 | 9.36 ± 6.78 |
| UNETR | 81.73 ± 13.26 | 71.09 ± 17.68 | 80.26 ± 14.66 | 83.42 ± 13.91 | 20.37 ± 21.52 | 11.71 ± 7.95 |
| Swin UNETR | 83.58 ± 12.31 | 73.56 ± 16.87 | 82.20 ± 13.54 | 85.16 ± 12.94 | 18.34 ± 19.87 | 7.03 ± 7.31 |
| MedNeXt | 85.42 ± 11.12 | 76.07 ± 15.71 | 84.06 ± 12.48 | 87.01 ± 11.73 | 16.03 ± 17.81 | 6.18 ± 6.46 |
| SegMamba | 86.13 ± 10.64 | 77.07 ± 15.21 | 84.91 ± 11.96 | 87.55 ± 11.21 | 17.37 ± 16.94 | 6.91 ± 6.17 |
| **VCAU-Net** | **87.54 ± 9.69** | **79.08 ± 14.21** | **86.42 ± 10.98** | **88.84 ± 10.27** | **14.18 ± 15.62** | **4.42 ± 5.71** |

### Independent external test cohort

| Method | Dice (%) | IoU (%) | Sensitivity (%) | Precision (%) | HD95 (mm) | ASSD (mm) |
|---|---:|---:|---:|---:|---:|---:|
| 3D U-Net | 71.86 ± 18.42 | 59.21 ± 21.37 | 70.44 ± 19.66 | 73.52 ± 18.93 | 31.47 ± 32.18 | 10.82 ± 12.91 |
| V-Net | 72.64 ± 17.96 | 60.13 ± 20.94 | 71.18 ± 19.08 | 74.35 ± 18.42 | 30.18 ± 30.96 | 10.31 ± 12.26 |
| nnU-Net v2 | 78.73 ± 14.86 | 66.73 ± 18.92 | 77.16 ± 16.24 | 80.55 ± 15.32 | 23.14 ± 24.61 | 7.46 ± 9.08 |
| UNETR | 75.36 ± 16.47 | 62.84 ± 20.06 | 73.74 ± 17.71 | 77.25 ± 16.93 | 27.23 ± 28.19 | 9.04 ± 10.73 |
| Swin UNETR | 77.14 ± 15.38 | 64.89 ± 19.34 | 75.63 ± 16.66 | 78.89 ± 15.87 | 24.91 ± 26.47 | 8.17 ± 9.82 |
| MedNeXt | 79.62 ± 14.21 | 67.82 ± 18.37 | 78.08 ± 15.61 | 81.40 ± 14.72 | 21.88 ± 23.16 | 7.02 ± 8.51 |
| SegMamba | 80.41 ± 13.72 | 68.86 ± 17.86 | 78.94 ± 14.95 | 82.11 ± 14.23 | 20.73 ± 22.08 | 6.63 ± 8.07 |
| **VCAU-Net** | **82.37 ± 12.86** | **71.49 ± 16.94** | **81.03 ± 14.11** | **83.94 ± 13.36** | **18.96 ± 20.37** | **5.91 ± 7.26** |

> **Note:** The manuscript is undergoing final statistical and implementation verification. Repository results will be synchronized with the accepted paper.

---

## Repository status

This repository currently serves as a **project and publication placeholder** while the manuscript is under peer review.

The following resources are planned for release after paper acceptance and completion of the final repository audit:

- [ ] Full VCAU-Net training code
- [ ] Inference code
- [ ] Pretrained model weights
- [ ] Five-fold checkpoints
- [ ] Data preprocessing scripts
- [ ] Rigid registration pipeline
- [ ] Dataset organization examples
- [ ] Training configuration files
- [ ] Inference configuration files
- [ ] Evaluation scripts
- [ ] Lesion-volume subgroup analysis
- [ ] Uncertainty and calibration analysis
- [ ] Computational profiling scripts
- [ ] Visualization scripts
- [ ] Reproduction instructions
- [ ] Environment specification

> **Code, pretrained weights, preprocessing scripts, and associated configuration files will be made publicly available after acceptance of the manuscript.**

---

## Planned repository structure

```text
VCAU-Net/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── train/
│   ├── inference/
│   └── datasets/
├── vcau_net/
│   ├── models/
│   │   ├── vcau_net.py
│   │   ├── spca.py
│   │   └── ugcf.py
│   ├── losses/
│   ├── data/
│   ├── transforms/
│   ├── inference/
│   └── evaluation/
├── scripts/
│   ├── preprocess/
│   ├── train/
│   ├── infer/
│   └── evaluate/
├── pretrained/
├── docs/
│   └── figures/
└── examples/
```

The final directory structure may change when the implementation is released.

---

## Data availability

### Internal cohort

The internal clinical dataset contains protected patient information and cannot be publicly distributed. Any future access will be subject to institutional ethics, privacy requirements, and data-use agreements.

### External cohort

The independent external evaluation uses the publicly available breast DCE-MRI dataset released by Zhang *et al.* and hosted on Zenodo.

**Dataset DOI:** `10.5281/zenodo.8068383`

Please cite the original dataset and associated publication when using these data.

---

## Preprocessing

The final preprocessing pipeline will be released after manuscript acceptance.

The planned release will document:

- enhancement-phase selection;
- image orientation handling;
- rigid cross-phase registration;
- voxel-spacing normalization;
- intensity normalization;
- image and label resampling;
- patch generation;
- augmentation;
- sliding-window inference settings.

No preprocessing code is publicly available in the current placeholder version.

---

## Model weights

Pretrained VCAU-Net weights are **not yet publicly available**.

The planned release includes:

- five fold-specific checkpoints;
- recommended inference weights;
- model configuration files;
- checksum information;
- example inference commands.

---

## Installation

Installation instructions will be added together with the public code release.

Expected core dependencies include:

```text
Python
PyTorch
MONAI
SimpleITK
NumPy
SciPy
scikit-image
```

Exact package versions will be provided in the final environment specification.

---

## Training

Training scripts and configuration files are currently withheld during peer review.

Planned usage:

```bash
# Placeholder command — not yet available
python scripts/train/train.py --config configs/train/vcau_net.yaml
```

---

## Inference

Inference code will be released together with the pretrained weights.

Planned usage:

```bash
# Placeholder command — not yet available
python scripts/infer/predict.py     --config configs/inference/vcau_net.yaml     --checkpoint pretrained/vcau_net_fold0.pth     --input <case_directory>     --output <output_directory>
```

The commands above are placeholders and may change before the official release.

---

## Evaluation

The public release will include scripts for:

- Dice
- IoU
- sensitivity
- precision
- HD95
- ASSD
- lesion-volume stratification
- ECE
- Brier score
- predictive-entropy analysis
- cross-center performance comparison

---

## Citation

The manuscript is currently under review. Citation information will be updated after publication.

```bibtex
@article{zhao_vcaunet,
  title   = {VCAU-Net: Structure-Preserving Cross-Phase Alignment and Uncertainty-Guided Fusion for 3D Breast Tumor Segmentation in DCE-MRI},
  author  = {Zhao, Sichao and Feng, Kanghua and Chen, Junjun and Su, Luowei and Li, Minghao and Zheng, Kehong and Lai, Wanting and Gao, Rong and Li, Weidong and Liu, Ying and Qiu, Xuejun},
  journal = {Under Review},
  year    = {2026}
}
```

---

## License

The license will be finalized together with the public code release.

Until then, this repository is provided for academic project display only. Unreleased code, model weights, and restricted clinical data are not licensed for redistribution unless explicitly authorized.

---

## Contact

For questions about this work, please open a GitHub issue after the public release or contact the corresponding author listed in the manuscript.

---

## Release plan

| Resource | Status |
|---|---|
| Paper | Under review |
| Source code | Planned after acceptance |
| Pretrained weights | Planned after acceptance |
| Preprocessing pipeline | Planned after acceptance |
| Training configurations | Planned after acceptance |
| Inference scripts | Planned after acceptance |
| Evaluation scripts | Planned after acceptance |
| Internal clinical data | Restricted |
| External public data | Available from the original dataset repository |

---

<p align="center">
  <b>VCAU-Net</b><br>
  Structure-preserving alignment before reliability-guided cross-phase fusion.
</p>
