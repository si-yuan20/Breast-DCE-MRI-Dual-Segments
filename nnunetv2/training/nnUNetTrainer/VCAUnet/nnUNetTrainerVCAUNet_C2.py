"""VCAU-Net ablation: C2 phase only.

The data pipeline is byte-identical to the full model -- the two-channel input
is unchanged and only the C5 branch is left unused. That keeps the input volume,
normalisation and augmentation exactly the same across every ablation arm, so
the only variable is what the network does with the two phases.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerVCAUNet import (
    nnUNetTrainerVCAUNet as _VCAUNetBase,
)


class nnUNetTrainerVCAUNet_C2(_VCAUNetBase):
    """Trained on the C2 phase only (channel 0)."""

    INPUT_MODE = "c2"
