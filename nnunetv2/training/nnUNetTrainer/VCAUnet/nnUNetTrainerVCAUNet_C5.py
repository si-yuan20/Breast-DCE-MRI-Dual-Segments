"""VCAU-Net ablation: C5 phase only.

See :mod:`nnunetv2.training.nnUNetTrainer.nnUNetTrainerVCAUNet_C2` for why the
input tensor is deliberately left at two channels.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerVCAUNet import (
    nnUNetTrainerVCAUNet as _VCAUNetBase,
)


class nnUNetTrainerVCAUNet_C5(_VCAUNetBase):
    """Trained on the C5 phase only (channel 1), where the tumour is enhanced."""

    INPUT_MODE = "c5"
