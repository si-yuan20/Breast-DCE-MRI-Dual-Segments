"""VCAU-Net v2: selective residual alignment + uncertainty-disagreement fusion.

Additive successor to :mod:`nnunetv2.nets.vcau`. The v1 package is kept
untouched for historical reproduction; v2 is a separate implementation with a
different stage policy, no state-space modelling, and a rebuilt fusion operator.

Pipeline::

    C2 -> Stem_C2 -> shared encoder -> F2[l]
    C5 -> Stem_C5 -> shared encoder -> F5[l]
      stages 2,3   : 3D-SPCA (bounded residual flow) -> UGCFv2
      stage  4     : UGCFv2 (no SPCA)
      stages 0,1,5 : lightweight cross-phase fusion
      -> nnU-Net UNetDecoder -> logits

See :mod:`nnunetv2.nets.vcau_v2.model` for the network and
:mod:`nnunetv2.training.nnUNetTrainer.nnUNetTrainerVCAUNetV2` for the trainer.
"""

from nnunetv2.nets.vcau_v2.model import (
    SPCA_STAGES,
    SPCA_MAX_OFFSETS,
    UGCF_STAGES,
    VCAUNetV2,
)

__all__ = ["VCAUNetV2", "SPCA_STAGES", "UGCF_STAGES", "SPCA_MAX_OFFSETS"]
