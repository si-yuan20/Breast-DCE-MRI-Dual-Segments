#!/usr/bin/env python3
"""
Export a trained nnU-Net checkpoint to ONNX for deployment and inference.

The exported graph is the SINGLE-PATCH forward pass:

    input  : (batch, C, D, H, W)  float32   -- C input channels, patch_size from plans
    output : (batch, num_classes, D, H, W)  float32 logits

Sliding-window tiling, test-time mirroring and the final softmax/argmax stay
on the deployment side (see the generated *.onnx.json sidecar for the exact
patch_size / spacing contract).

By default the ONNX file is written into an ``onnx/`` folder inside the
fold directory the checkpoint lives in, keeping a 1:1 name correspondence:

    {nnUNet_results}/DatasetXXX/...__...__3d_fullres/fold_N/
        checkpoint_best.pth
        onnx/
            checkpoint_best.onnx          <- exported graph
            checkpoint_best.onnx.json     <- metadata sidecar

Multiple checkpoints are exported in parallel (thread pool); each task is
dominated by C++ work (torch graph tracing, ONNX post-processing,
onnxruntime verification) which releases the GIL, so threads scale well.

Usage:
    python -m nnunetv2.inference.export_onnx -d 901 -tr nnUNetTrainer \
        -p nnUNetPlans -c 3d_fullres -f 1 --checkpoint checkpoint_best.pth

    python -m nnunetv2.inference.export_onnx \
        --model-dir /path/to/nnUNetTrainer__nnUNetPlans__3d_fullres \
        --fold 1 --all-checkpoints --num-workers 4

    python -m nnunetv2.inference.export_onnx -d Dataset901_Brain_ISLES2022 \
        -f 1 --checkpoint checkpoint_best.pth --output-dir /tmp/onnx
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import (
    join, isfile, subfiles,
)

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

ONNX_OPSET = 17
MAX_AUTO_WORKERS = 4


# ──────────────────────────────────────────────────────────────────
#  helpers
# ──────────────────────────────────────────────────────────────────

def _unwrap_network(predictor: nnUNetPredictor) -> torch.nn.Module:
    """Strip DDP / torch.compile wrappers from the predictor's network."""
    net = predictor.network
    if hasattr(net, 'module'):
        net = net.module
    if hasattr(net, '_orig_mod'):
        net = net._orig_mod
    return net


def _sha256(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _make_logger(prefix: str = None):
    """Thread-safe line logger; prefixes each line with the task name."""
    def log(*args):
        msg = ' '.join(str(a) for a in args)
        print(f'[{prefix}] {msg}' if prefix else msg, flush=True)
    return log


def _torch_onnx_export(network: torch.nn.Module, dummy: torch.Tensor,
                       onnx_path: str) -> None:
    """torch.onnx.export wrapper that prefers the stable TorchScript path.

    Newer PyTorch (>= 2.6) may default to the dynamo-based exporter, which
    requires the extra ``onnxscript`` package.  The TorchScript path
    (``dynamo=False``) is preferred: it only needs the ``onnx`` package and
    produces an equivalent inference graph.  Old torch versions without the
    ``dynamo`` argument fall through to the plain call.
    """
    import inspect
    export_kwargs = dict(
        input_names=['input'],
        output_names=['logits'],
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    )
    has_dynamo_flag = 'dynamo' in inspect.signature(torch.onnx.export).parameters
    try:
        if has_dynamo_flag:
            torch.onnx.export(network, dummy, onnx_path, dynamo=False, **export_kwargs)
        else:
            torch.onnx.export(network, dummy, onnx_path, **export_kwargs)
    except Exception as e:
        msg = str(e)
        if 'onnxscript' in msg:
            raise RuntimeError(
                "torch.onnx.export fell back to the dynamo exporter, which "
                "requires the 'onnxscript' package. Either install it "
                "('pip install onnxscript') or use a torch build where the "
                "TorchScript path (dynamo=False) is available."
            ) from e
        if 'onnx is not installed' in msg or "No module named 'onnx'" in msg:
            raise RuntimeError(
                "ONNX export requires the 'onnx' package for graph "
                "post-processing. Install it with: pip install onnx onnxruntime"
            ) from e
        raise


def _resolve_model_dir(args) -> str:
    """Return the model training output dir (…__plans__config level, no fold)."""
    if args.model_dir is not None:
        return args.model_dir
    # fold=None so get_output_folder stops before fold_N; fold_N is appended later
    return get_output_folder(args.dataset, args.trainer, args.plans,
                             args.configuration, fold=None)


def _build_predictor(model_dir: str, fold: int, checkpoint_name: str) -> nnUNetPredictor:
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device('cpu'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=model_dir,
        use_folds=(fold,),
        checkpoint_name=checkpoint_name,
    )
    return predictor


# ──────────────────────────────────────────────────────────────────
#  core export
# ──────────────────────────────────────────────────────────────────

def export_one(model_dir: str, fold: int, checkpoint_name: str,
               output_dir: str = None, overwrite: bool = False,
               verify: bool = True, log=None, ort_threads: int = None) -> dict:
    """Export one checkpoint to ONNX and return its metadata dict.

    Args:
        log: logging callable (default: print). In parallel mode a prefixed
             logger is passed so interleaved output stays readable.
        ort_threads: intra-op thread count for the onnxruntime verification
             session; None uses the onnxruntime default (all cores).
    """
    if log is None:
        log = print

    fold_dir = join(model_dir, f'fold_{fold}')
    ckpt_path = join(fold_dir, checkpoint_name)
    if not isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt_stem = checkpoint_name[:-4] if checkpoint_name.endswith('.pth') else checkpoint_name
    if output_dir is None:
        output_dir = join(fold_dir, 'onnx')
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = join(output_dir, ckpt_stem + '.onnx')
    meta_path = onnx_path + '.json'

    if os.path.isfile(onnx_path) and not overwrite:
        raise FileExistsError(
            f'{onnx_path} already exists. Pass --overwrite to replace it.')

    t_start = time.perf_counter()
    log(f'Loading checkpoint: {ckpt_path}')
    predictor = _build_predictor(model_dir, fold, checkpoint_name)
    network = _unwrap_network(predictor)
    network.eval()

    patch_size = tuple(int(s) for s in predictor.configuration_manager.patch_size)
    num_input_channels = determine_num_input_channels(
        predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json)
    num_classes = predictor.label_manager.num_segmentation_heads
    spacing = tuple(float(s) for s in predictor.configuration_manager.spacing)

    dummy = torch.randn(1, num_input_channels, *patch_size)
    with torch.no_grad():
        ref = network(dummy)
    if isinstance(ref, (list, tuple)):
        ref = ref[0]
    ref = ref.float()
    log(f'Input : (1, {num_input_channels}, {", ".join(map(str, patch_size))})')
    log(f'Output: {tuple(ref.shape)}  (num_classes={num_classes})')

    # ── dependency pre-check ──
    _has_onnx = True
    try:
        import onnx  # noqa: F401
    except ImportError:
        _has_onnx = False
        log('WARNING: the "onnx" package is not installed — export may fail.')
        log('         Install with: pip install onnx onnxruntime')

    # ── export ──
    log(f'Exporting to ONNX (opset {ONNX_OPSET}) ...')
    t_export = time.perf_counter()
    try:
        with torch.no_grad():
            _torch_onnx_export(network, dummy, onnx_path)
    except RuntimeError as e:
        if not _has_onnx:
            raise RuntimeError(
                'ONNX export needs the "onnx" package (and onnxruntime for '
                'verification). Install with: pip install onnx onnxruntime'
            ) from e
        raise
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    log(f'Written: {onnx_path} ({size_mb:.1f} MB, '
        f'export took {time.perf_counter() - t_export:.1f}s)')

    # ── verify with onnxruntime ──
    # Two-level check: (1) raw logit agreement, (2) argmax segmentation
    # agreement — the latter is what deployment actually consumes.  fp32
    # export may differ from eager by ~1e-3 in logits, which almost never
    # flips an argmax decision.
    verify_result = {'performed': False}
    if verify:
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if ort_threads is not None:
                so.intra_op_num_threads = ort_threads
            sess = ort.InferenceSession(onnx_path, sess_options=so,
                                        providers=['CPUExecutionProvider'])
            got = sess.run(['logits'], {'input': dummy.numpy()})[0]

            # dynamic batch check: run the same session with batch=2
            dummy2 = torch.randn(2, num_input_channels, *patch_size)
            with torch.no_grad():
                ref2 = network(dummy2)
            if isinstance(ref2, (list, tuple)):
                ref2 = ref2[0]
            ref2 = ref2.float().numpy()
            got2 = sess.run(['logits'], {'input': dummy2.numpy()})[0]

            max_diff = float(np.abs(ref.numpy() - got).max())
            max_diff2 = float(np.abs(ref2 - got2).max())

            seg_ref = ref.argmax(dim=1).numpy()
            seg_got = got.argmax(axis=1)
            agree = float((seg_ref == seg_got).mean())
            seg_ref2 = ref2.argmax(axis=1)
            seg_got2 = got2.argmax(axis=1)
            agree2 = float((seg_ref2 == seg_got2).mean())

            LOGIT_TOL = 1e-2      # fp32 export tolerance on raw logits
            AGREE_TOL = 0.999     # argmax segmentation must match ~exactly
            ok = (max_diff < LOGIT_TOL and max_diff2 < LOGIT_TOL
                  and agree >= AGREE_TOL and agree2 >= AGREE_TOL)
            verify_result = {
                'performed': True,
                'max_abs_diff_batch1': max_diff,
                'max_abs_diff_batch2': max_diff2,
                'seg_agreement_batch1': agree,
                'seg_agreement_batch2': agree2,
                'passed': ok,
            }
            status = 'PASS' if ok else 'FAIL'
            log(f'onnxruntime check: {status}')
            log(f'  batch1: max|Δ logits|={max_diff:.2e}  '
                f'argmax agreement={agree * 100:.4f}%')
            log(f'  batch2: max|Δ logits|={max_diff2:.2e}  '
                f'argmax agreement={agree2 * 100:.4f}%')
            if not ok:
                log('WARNING: verification thresholds exceeded — out of '
                    'tolerance (logits < 1e-2, argmax agreement >= 99.9%)')
        except ImportError:
            log('onnxruntime not installed — skipped numerical verification.')
            log('Install with: pip install onnxruntime')
            verify_result = {'performed': False, 'reason': 'onnxruntime not installed'}

    # ── metadata sidecar ──
    meta = {
        'source_checkpoint': ckpt_path,
        'checkpoint_sha256': _sha256(ckpt_path),
        'model_dir': model_dir,
        'fold': fold,
        'trainer_name': predictor.trainer_name,
        'input_channels': num_input_channels,
        'num_classes': num_classes,
        'patch_size': list(patch_size),
        'target_spacing': list(spacing),
        'input_shape_fixed': [1, num_input_channels, *patch_size],
        'dynamic_axes': {'batch': 0},
        'opset': ONNX_OPSET,
        'torch_version': torch.__version__,
        'export_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'export_seconds': round(time.perf_counter() - t_start, 1),
        'onnx_file': os.path.basename(onnx_path),
        'onnx_size_mb': round(size_mb, 2),
        'verification': verify_result,
        'deployment_note': (
            'Feed (batch, C, D, H, W) float32 patches of exactly patch_size; '
            'apply sliding-window assembly, mirroring TTA and softmax/argmax '
            'on the deployment side. Spacing must match target_spacing.'
        ),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f'Metadata: {meta_path}')
    log(f'DONE in {time.perf_counter() - t_start:.1f}s')

    return meta


def _export_task(model_dir, fold, ckpt, output_dir, overwrite, verify, ort_threads):
    """Thread-pool task wrapper: never raises, returns (ckpt, meta, error)."""
    log = _make_logger(ckpt)
    log(f'--- start ---')
    try:
        meta = export_one(model_dir, fold, ckpt,
                          output_dir=output_dir, overwrite=overwrite,
                          verify=verify, log=log, ort_threads=ort_threads)
        return ckpt, meta, None
    except Exception as e:
        import traceback
        log(f'FAILED: {e}')
        return ckpt, None, f'{e}\n{traceback.format_exc()}'


# ──────────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Export nnU-Net checkpoints to ONNX (single-patch forward graph)')
    parser.add_argument('-d', '--dataset', type=str, default=None,
                        help='Dataset name or id (e.g. 901 or Dataset901_Brain_ISLES2022)')
    parser.add_argument('-tr', '--trainer', type=str, default='nnUNetTrainer')
    parser.add_argument('-p', '--plans', type=str, default='nnUNetPlans')
    parser.add_argument('-c', '--configuration', type=str, default='3d_fullres')
    parser.add_argument('-f', '--fold', type=int, default=0)
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Direct path to the model training output directory '
                             '(overrides -d/-tr/-p/-c addressing)')
    parser.add_argument('--checkpoint', nargs='+', default=None,
                        help='Checkpoint filename(s) inside fold_N/ '
                             '(default: checkpoint_best.pth)')
    parser.add_argument('--all-checkpoints', action='store_true',
                        help='Export every checkpoint_*.pth in the fold directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: onnx/ inside the '
                             'fold directory next to the checkpoint)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Parallel export workers for multiple checkpoints '
                             f'(default: auto = min(n_checkpoints, cpu_count, {MAX_AUTO_WORKERS}); '
                             'use 1 for sequential)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing ONNX files')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip the onnxruntime numerical verification')
    args = parser.parse_args()

    if args.model_dir is None and args.dataset is None:
        parser.error('Provide either --dataset or --model-dir')

    model_dir = _resolve_model_dir(args)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f'Model directory not found: {model_dir}')

    fold_dir = join(model_dir, f'fold_{args.fold}')
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f'Fold directory not found: {fold_dir}')

    # resolve checkpoint list
    if args.all_checkpoints:
        checkpoints = [os.path.basename(p)
                       for p in subfiles(fold_dir, suffix='.pth', join=True)]
        if not checkpoints:
            raise FileNotFoundError(f'No .pth checkpoints found in {fold_dir}')
        checkpoints = sorted(checkpoints)
    elif args.checkpoint:
        checkpoints = args.checkpoint
    else:
        checkpoints = ['checkpoint_best.pth']

    # worker count
    cpu = os.cpu_count() or 1
    if args.num_workers is None:
        n_workers = max(1, min(len(checkpoints), cpu, MAX_AUTO_WORKERS))
    else:
        n_workers = max(1, min(args.num_workers, len(checkpoints)))
    # limit onnxruntime intra-op threads per session when running in parallel
    ort_threads = None if n_workers == 1 else max(1, cpu // n_workers)

    print('=' * 70)
    print(f'  ONNX export')
    print(f'  model_dir : {model_dir}')
    print(f'  fold      : {args.fold}')
    print(f'  checkpoints: {len(checkpoints)}')
    print(f'  workers   : {n_workers}'
          + (f' (ort intra-op threads per session: {ort_threads})' if ort_threads else ''))
    print(f'  started   : {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70, flush=True)

    t_main = time.perf_counter()
    results = []

    if n_workers > 1:
        # keep torch intra-op threads from oversubscribing during parallel traces
        try:
            torch.set_num_threads(max(1, cpu // n_workers))
        except Exception:
            pass
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(_export_task, model_dir, args.fold, ckpt,
                          args.output_dir, args.overwrite,
                          not args.no_verify, ort_threads): ckpt
                for ckpt in checkpoints
            }
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for ckpt in checkpoints:
            results.append(_export_task(model_dir, args.fold, ckpt,
                                        args.output_dir, args.overwrite,
                                        not args.no_verify, ort_threads))

    elapsed = time.perf_counter() - t_main

    # ── summary (original checkpoint order) ──
    order = {c: i for i, c in enumerate(checkpoints)}
    results.sort(key=lambda r: order.get(r[0], 1 << 30))

    print('\n' + '=' * 70)
    print('  Export summary')
    print('=' * 70)
    n_ok = sum(1 for _, m, _ in results if m is not None)
    for ckpt, meta, err in results:
        if meta is not None:
            v = meta['verification']
            v_status = ('PASS' if v.get('passed') else
                        ('skipped' if not v.get('performed') else 'FAIL'))
            print(f"  [OK]   {ckpt:<45} {meta['onnx_size_mb']:>7.1f} MB  "
                  f"verify={v_status:<7} {meta['export_seconds']:>6.1f}s")
        else:
            first_line = err.splitlines()[0] if err else 'unknown error'
            print(f'  [FAIL] {ckpt:<45} {first_line}')
    print(f'\n  {n_ok}/{len(results)} exported successfully '
          f'in {elapsed:.1f}s total.')
    print('=' * 70)


if __name__ == '__main__':
    main()
