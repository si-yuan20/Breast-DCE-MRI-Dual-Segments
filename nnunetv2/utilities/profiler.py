"""
Lightweight training profiler for nnU-Net stall diagnosis.

Activate via environment variable:
    NNUNET_PROFILE=1 nnUNetv2_train ...

Deactivate (default): profiling calls become no-ops.
"""
import os
import time
import torch
import functools
import warnings
from collections import defaultdict
from typing import Optional

# ── environment flag ──────────────────────────────────────────────────
_ENABLED = os.environ.get("NNUNET_PROFILE", "0").lower() in ("1", "true", "t")

# ── deduplicated warning filter ───────────────────────────────────────
_warned_msgs = set()


def _warn_once(msg: str, category=UserWarning) -> None:
    """Emit a warning only the first time a given message is seen."""
    if msg not in _warned_msgs:
        _warned_msgs.add(msg)
        warnings.warn(msg, category)


# ── GPU memory snapshot ────────────────────────────────────────────────
class GPUMemSnapshot:
    """Capture and compare GPU memory usage before/after an operation."""

    __slots__ = ("allocated", "reserved", "max_allocated", "timestamp")

    def __init__(self):
        self.allocated = 0
        self.reserved = 0
        self.max_allocated = 0
        self.timestamp = 0.0

    @staticmethod
    def take() -> "GPUMemSnapshot":
        s = GPUMemSnapshot()
        if torch.cuda.is_available():
            s.allocated = torch.cuda.memory_allocated()
            s.reserved = torch.cuda.memory_reserved()
            s.max_allocated = torch.cuda.max_memory_allocated()
            s.timestamp = time.time()
        return s

    def diff_str(self, baseline: "GPUMemSnapshot") -> str:
        d_alloc = (self.allocated - baseline.allocated) / 1024**2
        d_resv = (self.reserved - baseline.reserved) / 1024**2
        alloc_mb = self.allocated / 1024**2
        resv_mb = self.reserved / 1024**2
        max_mb = self.max_allocated / 1024**2
        return (
            f"alloc={alloc_mb:.0f}MB(Δ{d_alloc:+.0f}MB) "
            f"reserved={resv_mb:.0f}MB(Δ{d_resv:+.0f}MB) "
            f"max_alloc={max_mb:.0f}MB"
        )


# ── simple timer ───────────────────────────────────────────────────────
class Timer:
    """Wall-clock timer with label."""

    def __init__(self, label: str, print_fn=None):
        self.label = label
        self.print_fn = print_fn or (lambda msg: print(msg, flush=True))
        self.t0 = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        if _ENABLED:
            torch.cuda.synchronize()
            self.t0 = time.time()
            mem = GPUMemSnapshot.take()
            self.print_fn(
                f"[PROFILE:BEGIN] {self.label} "
                f"| GPU: {mem.diff_str(GPUMemSnapshot())}"
            )
        return self

    def __exit__(self, *args):
        if _ENABLED:
            torch.cuda.synchronize()
            self.elapsed = time.time() - self.t0
            mem = GPUMemSnapshot.take()
            self.print_fn(
                f"[PROFILE:END]   {self.label} "
                f"| cost={self.elapsed:.3f}s "
                f"| GPU: alloc={mem.allocated/1024**2:.0f}MB "
                f"reserved={mem.reserved/1024**2:.0f}MB"
            )


# ── per-epoch profiling context ────────────────────────────────────────
class EpochProfiler:
    """Tracks timing and GPU memory across one epoch."""

    __slots__ = (
        "epoch",
        "print_fn",
        "cumulative",
        "gpu_history",
        "epoch_start",
    )

    def __init__(self, epoch: int, print_fn=None):
        self.epoch = epoch
        self.print_fn = print_fn or print
        self.cumulative: dict = defaultdict(float)
        self.gpu_history: list = []
        self.epoch_start = GPUMemSnapshot.take() if _ENABLED else None

    def record(self, key: str, seconds: float):
        self.cumulative[key] += seconds

    def record_gpu(self):
        if _ENABLED and torch.cuda.is_available():
            self.gpu_history.append(GPUMemSnapshot.take())

    def summary(self, epoch_time_s: float) -> str:
        if not _ENABLED:
            return ""
        lines = [f"\n[PROFILE] ══ Epoch {self.epoch} summary ══"]
        lines.append(f"  Total epoch time: {epoch_time_s:.2f}s")
        for k, v in sorted(self.cumulative.items(), key=lambda x: -x[1]):
            pct = v / epoch_time_s * 100 if epoch_time_s > 0 else 0
            lines.append(f"  {k:40s} {v:8.2f}s  ({pct:5.1f}%)")
        if self.gpu_history:
            first = self.gpu_history[0]
            last = self.gpu_history[-1]
            lines.append(f"  GPU start: {first.diff_str(GPUMemSnapshot())}")
            lines.append(f"  GPU end:   {last.diff_str(GPUMemSnapshot())}")
        lines.append("═" * 50)
        return "\n".join(lines)


# ── convenience helpers ────────────────────────────────────────────────
def is_enabled() -> bool:
    return _ENABLED


def profile_gpu_mem(print_fn=None):
    """One-shot GPU memory snapshot to stdout / log."""
    if not _ENABLED or not torch.cuda.is_available():
        return
    s = GPUMemSnapshot.take()
    msg = (
        f"[GPU MEM] allocated={s.allocated/1024**2:.0f}MB "
        f"reserved={s.reserved/1024**2:.0f}MB "
        f"max_allocated={s.max_allocated/1024**2:.0f}MB"
    )
    if print_fn:
        print_fn(msg)
    else:
        print(msg, flush=True)


def reset_peak_memory():
    """Reset CUDA peak memory stats (call at epoch start)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def timed_block(key: str, epoch_profiler: Optional[EpochProfiler] = None):
    """Context manager that records wall-clock time.

    Usage:
        with timed_block("save_checkpoint", epoch_profiler) as t:
            ...
        # t.elapsed is available after exit
    """
    class _TimedCtx:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            torch.cuda.synchronize()
            self._t0 = time.time()
            return self
        def __exit__(self, *args):
            torch.cuda.synchronize()
            self.elapsed = time.time() - self._t0
            if epoch_profiler is not None:
                epoch_profiler.record(key, self.elapsed)
            if _ENABLED and self.elapsed > 5.0:
                print(f"[PROFILE:SLOW] {key} took {self.elapsed:.2f}s", flush=True)
    return _TimedCtx()
