"""VCAU-Net ablation: alignment without uncertainty-guided fusion.

3D-SPCA, the auxiliary reliability heads and the alignment loss are all kept;
only the fusion operator changes from reliability-weighted to plain
concatenation of the aligned features. The reliability maps are still computed
and still drive the alignment weighting, so this arm isolates the fusion
weighting itself.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerVCAUNet import (
    nnUNetTrainerVCAUNet as _VCAUNetBase,
)


class nnUNetTrainerVCAUNet_NoUGCF(_VCAUNetBase):
    """Reliability-guided fusion replaced by plain concatenation."""

    FUSION_MODE = "direct"
