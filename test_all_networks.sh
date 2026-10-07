#!/bin/bash
# ==============================================================================
# test_all_networks.sh
# ==============================================================================
# Test all custom Trainer networks for nnU-Net v2 on Dataset305, fold 0, 4 GPUs.
#
# Usage:
#   bash test_all_networks.sh                          # All trainers, quick test
#   bash test_all_networks.sh --quick                  # Import + dry-forward only
#   bash test_all_networks.sh --full                   # Full 50-epoch test
#   bash test_all_networks.sh --trainer SegResNet      # Single trainer test
#   bash test_all_networks.sh --list                   # List available trainers
#   bash test_all_networks.sh --diagnose               # 1-epoch diagnose with debug logs
#
# Configuration:
#   DATASET=305, FOLD=0, GPUS=4, CONFIG=3d_fullres
# ==============================================================================

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
DATASET="${DATASET:-305}"
FOLD="${FOLD:-0}"
GPUS="${GPUS:-4}"
CONFIG="${CONFIG:-3d_fullres}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
QUICK_EPOCHS=5
DIAGNOSE_EPOCHS=1
FULL_EPOCHS=50
LOG_DIR="${PROJECT_ROOT}/test_logs_$(date +%Y%m%d_%H%M%S)"

# ── Trainer list ─────────────────────────────────────────────────────────────
ALL_TRAINERS=(
    "nnUNetTrainer"                          # base control
    "nnUNetTrainerUNETR"
    "nnUNetTrainerSwinUNETR"
    "nnUNetTrainerSegResNet"
    "nnUNetTrainerVNet"
    "nnUNetTrainerDynUNet"
    "nnUNetTrainerAttentionUnet"
    "nnUNetTrainerSegFormer"
    "nnUNetTrainerSegMamba"
    "nnUNetTrainerUMamba"
    "nnUNetTrainerVeloxSeg"
    "nnUNetTrainerLMambaMFDSNet"
    "nnUNetTrainerBaseChainLMambaMFDS"
)

# ── Colors ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_pass()  { echo -e "${GREEN}[PASS]${NC} $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
log_sep()   { echo "════════════════════════════════════════════════════════════"; }

# ── Check prerequisites ──────────────────────────────────────────────────────
check_prereqs() {
    log_info "Checking prerequisites..."
    mkdir -p "${LOG_DIR}"

    if ! command -v python &>/dev/null; then
        log_fail "Python not found"
        exit 1
    fi

    python -c "
import torch
assert torch.cuda.device_count() >= ${GPUS}, f'Need ${GPUS} GPUs, found {torch.cuda.device_count()}'
print(f'CUDA OK: {torch.cuda.device_count()} GPUs')
" 2>&1 | tee "${LOG_DIR}/00_prereqs.log"
    log_pass "Prerequisites OK"
}

# ── Import check (fast, no GPU needed) ───────────────────────────────────────
import_check() {
    local trainer=$1
    local logfile="${LOG_DIR}/import_${trainer}.log"

    python -c "
from nnunetv2.training.nnUNetTrainer.${trainer} import ${trainer}
print('Import OK:', ${trainer}.__name__)
print('Bases:', [b.__name__ for b in ${trainer}.__mro__])
" > "${logfile}" 2>&1 && log_pass "Import ${trainer}" || log_fail "Import ${trainer} (see ${logfile})"
}

# ── Dry forward check (builds network, no training) ──────────────────────────
dry_forward_check() {
    local trainer=$1
    local logfile="${LOG_DIR}/dryforward_${trainer}.log"

    python -c "
import torch
import numpy as np
from nnunetv2.training.nnUNetTrainer.${trainer} import ${trainer}
from nnunetv2.paths import nnUNet_preprocessed
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

dataset_name = maybe_convert_to_dataset_name(${DATASET})
pp_folder = join(nnUNet_preprocessed, dataset_name)
plans = load_json(join(pp_folder, 'nnUNetPlans.json'))
plans['continue_training'] = False
dataset_json = load_json(join(pp_folder, 'dataset.json'))

device = torch.device('cuda:0')
t = ${trainer}(plans=plans, configuration='${CONFIG}', fold=${FOLD}, dataset_json=dataset_json, device=device)
t.initialize()

model = t.network
if hasattr(model, 'module'):
    model = model.module

params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {params:,} total / {trainable:,} trainable')

# Check BatchNorm3d
bn_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm3d))
print(f'BatchNorm3d: {bn_count} (should be 0)')

# Dry forward
in_ch = t.num_input_channels
ps = t.configuration_manager.patch_size
x = torch.randn(1, in_ch, *ps).to(device)
with torch.no_grad():
    out = model(x)

if isinstance(out, (list, tuple)):
    print(f'Output: list of {len(out)} tensors')
    for i, o in enumerate(out):
        print(f'  [{i}]: shape={tuple(o.shape)}, nan={torch.isnan(o).any().item()}, inf={torch.isinf(o).any().item()}')
    out0 = out[0]
else:
    print(f'Output: single tensor shape={tuple(out.shape)}')
    out0 = out

spatial_match = (tuple(out0.shape[2:]) == tuple(ps))
print(f'Spatial match: {spatial_match} (out={tuple(out0.shape[2:])}, patch={tuple(ps)})')
print(f'DS enabled: {t.enable_deep_supervision}')
print(f'Compile: {t._do_i_compile()}')
print(f'Init LR: {t.initial_lr}, WD: {t.weight_decay}')
print('DRY FORWARD OK')

del model, x, out
torch.cuda.empty_cache()
" > "${logfile}" 2>&1 && log_pass "DryFwd  ${trainer}" || log_fail "DryFwd  ${trainer} (see ${logfile})"
}

# ── Quick training (N epochs) ────────────────────────────────────────────────
quick_train() {
    local trainer=$1
    local epochs=$2
    local label=$3
    local logfile="${LOG_DIR}/train_${label}_${trainer}.log"

    log_info "Training ${trainer} for ${epochs} epochs (${label})..."
    export NNUNET_CUSTOM_DEBUG=1
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

    python -c "
import os, sys, torch, torch.distributed as dist
from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '${CUDA_DEVICES}'

trainer = get_trainer_from_args('${DATASET}', '${CONFIG}', ${FOLD}, '${trainer}')
trainer.num_epochs = ${epochs}
trainer.save_every = ${epochs}  # save only at end

trainer.on_train_start()
try:
    for epoch in range(${epochs}):
        trainer.on_epoch_start()
        trainer.on_train_epoch_start()
        train_outputs = []
        for _ in range(trainer.num_iterations_per_epoch):
            train_outputs.append(trainer.train_step(next(trainer.dataloader_train)))
        trainer.on_train_epoch_end(train_outputs)
        with torch.no_grad():
            trainer.on_validation_epoch_start()
            val_outputs = []
            for _ in range(trainer.num_val_iterations_per_epoch):
                val_outputs.append(trainer.validation_step(next(trainer.dataloader_val)))
            trainer.on_validation_epoch_end(val_outputs)
        trainer.on_epoch_end()
    trainer.on_train_end()
    print('TRAINING COMPLETE OK')
except Exception as e:
    print(f'TRAINING FAILED: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" > "${logfile}" 2>&1 && log_pass "Train   ${trainer} (${label})" || log_fail "Train   ${trainer} (${label}) (see ${logfile})"
}

# ── Diagnose mode (1 epoch, verbose debug) ───────────────────────────────────
diagnose_trainer() {
    local trainer=$1
    local logfile="${LOG_DIR}/diagnose_${trainer}.log"

    log_info "Diagnosing ${trainer} (1 epoch, full debug)..."
    export NNUNET_CUSTOM_DEBUG=1
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

    python -c "
import os, sys, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '${CUDA_DEVICES}'

from nnunetv2.run.run_training import get_trainer_from_args
trainer = get_trainer_from_args('${DATASET}', '${CONFIG}', ${FOLD}, '${trainer}')
trainer.num_epochs = 1
trainer.save_every = 100

# Print all key configs
print('='*60)
print(f'Trainer: ${trainer}')
print(f'DS enabled: {trainer.enable_deep_supervision}')
print(f'DS scales: {trainer._get_deep_supervision_scales()}')
print(f'Compile: {trainer._do_i_compile()}')
print(f'LR={trainer.initial_lr} WD={trainer.weight_decay}')
print(f'Patch: {trainer.configuration_manager.patch_size}')
print(f'Batch: {trainer.configuration_manager.batch_size}')
print(f'Features: {trainer.configuration_manager.UNet_base_num_features}->{trainer.configuration_manager.unet_max_num_features}')
print('='*60)

trainer.on_train_start()

for epoch in range(1):
    trainer.on_epoch_start()
    trainer.on_train_epoch_start()
    train_outputs = []
    for i in range(min(5, trainer.num_iterations_per_epoch)):
        out = trainer.train_step(next(trainer.dataloader_train))
        train_outputs.append(out)
        if i == 0:
            print(f'[Iter 0] loss={out[\"loss\"]:.4f}')
    trainer.on_train_epoch_end(train_outputs)

    with torch.no_grad():
        trainer.on_validation_epoch_start()
        val_outputs = []
        for i in range(min(5, trainer.num_val_iterations_per_epoch)):
            out = trainer.validation_step(next(trainer.dataloader_val))
            val_outputs.append(out)
            if i == 0:
                print(f'[Val 0] loss={out[\"loss\"]:.4f}')

        # Extra diagnotics on first val batch
        batch = next(trainer.dataloader_val)
        data = batch['data'].to(trainer.device)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(trainer.device) for t in target]
        else:
            target = target.to(trainer.device)
        with torch.no_grad():
            output = trainer.network(data)
        if isinstance(output, list):
            out0 = output[0]
            print(f'Output: list[{len(output)}], shape[0]={tuple(out0.shape)}')
        else:
            out0 = output
            print(f'Output: single, shape={tuple(out0.shape)}')
        pred = out0.argmax(1)
        uniq, cnt = pred.unique(return_counts=True)
        print(f'Pred classes: {dict(zip(uniq.tolist(), cnt.tolist()))}')
        probs = torch.softmax(out0, dim=1)
        for c in range(1, out0.shape[1]):
            print(f'  Class {c} prob: min={probs[:,c].min():.4f} max={probs[:,c].max():.4f} mean={probs[:,c].mean():.4f}')

    trainer.on_validation_epoch_end(val_outputs)
    trainer.on_epoch_end()
    print(f'Train loss: {trainer.logger.get_value(\"train_losses\", step=-1):.4f}')
    print(f'Val loss:   {trainer.logger.get_value(\"val_losses\", step=-1):.4f}')
    print(f'PseudoDice: {trainer.logger.get_value(\"dice_per_class_or_region\", step=-1)}')

trainer.on_train_end()
print('DIAGNOSE COMPLETE')
" > "${logfile}" 2>&1 && log_pass "Diagnose ${trainer}" || log_fail "Diagnose ${trainer} (see ${logfile})"
}

# ── Main ─────────────────────────────────────────────────────────────────────

main_quick() {
    log_sep
    log_info "MODE: Quick (import + dry-forward only)"
    log_sep
    check_prereqs

    for trainer in "${ALL_TRAINERS[@]}"; do
        log_sep
        import_check "${trainer}"
        dry_forward_check "${trainer}"
    done
}

main_diagnose() {
    log_sep
    log_info "MODE: Diagnose (1 epoch per trainer)"
    log_sep
    check_prereqs

    for trainer in "${ALL_TRAINERS[@]}"; do
        log_sep
        diagnose_trainer "${trainer}"
    done
}

main_full() {
    log_sep
    log_info "MODE: Full (${FULL_EPOCHS} epochs per trainer)"
    log_sep
    check_prereqs

    for trainer in "${ALL_TRAINERS[@]}"; do
        log_sep
        import_check "${trainer}"
        quick_train "${trainer}" "${FULL_EPOCHS}" "full"
    done
}

main_single() {
    local trainer=$1
    log_sep
    log_info "MODE: Single trainer (${trainer})"
    log_sep
    check_prereqs
    import_check "${trainer}"
    dry_forward_check "${trainer}"
    diagnose_trainer "${trainer}"
}

main_list() {
    echo "Available trainers:"
    for i in "${!ALL_TRAINERS[@]}"; do
        printf "  %2d. %s\n" "$((i+1))" "${ALL_TRAINERS[$i]}"
    done
}

# ── Argument parsing ─────────────────────────────────────────────────────────

MODE="diagnose"  # default

if [[ $# -eq 0 ]]; then
    MODE="diagnose"
else
    case "$1" in
        --quick)   MODE="quick" ;;
        --full)    MODE="full" ;;
        --diagnose) MODE="diagnose" ;;
        --list)    MODE="list" ;;
        --trainer)
            if [[ -z "${2:-}" ]]; then
                echo "Usage: $0 --trainer <TrainerName>"
                exit 1
            fi
            MODE="single"
            SINGLE_TRAINER="$2"
            ;;
        *)
            echo "Usage: $0 [--quick|--full|--diagnose|--list|--trainer <Name>]"
            echo "  --quick     Import + dry-forward only (fast)"
            echo "  --diagnose  1-epoch diagnose with verbose debug (default)"
            echo "  --full      50-epoch training test"
            echo "  --list      List available trainers"
            echo "  --trainer   Test single trainer"
            exit 1
            ;;
    esac
fi

# ── Execute ──────────────────────────────────────────────────────────────────

case "${MODE}" in
    quick)    main_quick ;;
    full)     main_full ;;
    diagnose) main_diagnose ;;
    list)     main_list ;;
    single)   main_single "${SINGLE_TRAINER}" ;;
esac

# ── Summary ──────────────────────────────────────────────────────────────────
log_sep
log_info "All logs saved to: ${LOG_DIR}"

PASS_COUNT=$(grep -l "PASS\|OK\|COMPLETE" "${LOG_DIR}"/*.log 2>/dev/null | wc -l)
FAIL_COUNT=$(grep -l "FAIL\|Error\|Traceback" "${LOG_DIR}"/*.log 2>/dev/null | wc -l)

echo ""
echo "Summary:"
echo "  Pass: ${PASS_COUNT}"
echo "  Fail: ${FAIL_COUNT}"
echo "  Logs: ${LOG_DIR}"
echo ""
echo "To view failures:"
echo "  grep -l 'FAIL\|Error' ${LOG_DIR}/*.log"
echo ""
echo "To view a specific log:"
echo "  cat ${LOG_DIR}/diagnose_nnUNetTrainerSegResNet.log"
