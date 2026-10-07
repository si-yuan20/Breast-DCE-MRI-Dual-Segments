"""VCAU-Net ablation: naive C2+C5 fusion (neither 3D-SPCA nor UGCF).

The reference dual-phase baseline: both phases are encoded by the shared
encoder and concatenated by a plain 1x1x1 convolution, with no alignment and no
reliability weighting. Every other dual-phase arm is measured against this one.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerVCAUNet import (
    nnUNetTrainerVCAUNet as _VCAUNetBase,
)


class nnUNetTrainerVCAUNet_Direct(_VCAUNetBase):
    """Dual-phase input with plain concatenation fusion."""

    FUSION_MODE = "direct"
    USE_SPCA = False
