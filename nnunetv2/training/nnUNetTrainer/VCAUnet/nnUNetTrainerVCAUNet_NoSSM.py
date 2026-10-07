"""VCAU-Net ablation: full alignment and fusion, no state-space bottleneck.

The fused bottleneck feeds the decoder directly instead of taking a detour
through the axis-wise bidirectional state-space block. This isolates the
long-range volumetric context term.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerVCAUNet import (
    nnUNetTrainerVCAUNet as _VCAUNetBase,
)


class nnUNetTrainerVCAUNet_NoSSM(_VCAUNetBase):
    """Volumetric state-space bottleneck disabled."""

    USE_SSM = False
