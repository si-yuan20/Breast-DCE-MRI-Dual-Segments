"""VCAU-Net ablation: uncertainty-guided fusion without 3D-SPCA.

The reliability branch and the fusion operator are untouched; only the bounded
residual alignment is removed. Fusion then sees unaligned C5 features, and the
structure alignment loss does not exist in this arm (there is no warp to
measure consistency after). This isolates what the cross-phase alignment alone
contributes on top of UGCF.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerVCAUNet import (
    nnUNetTrainerVCAUNet as _VCAUNetBase,
)


class nnUNetTrainerVCAUNet_NoSPCA(_VCAUNetBase):
    """3D-SPCA disabled; UGCF kept."""

    USE_SPCA = False
