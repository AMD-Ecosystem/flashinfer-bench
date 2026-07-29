"""
Timing utilities for benchmarking FlashInfer-Bench kernel solutions.

The timing backend is selectable via the ``FIB_TIMING_BACKEND`` environment variable:

- ``torch_events`` (default): portable GPU timing using ``torch.cuda.Event`` (backed by HIP events
  on ROCm / CUDA events on NVIDIA). Self-contained, no dependency on ``flashinfer.testing``.
- ``rocprof`` (reserved): device-side kernel timing via the ``rocprofv3`` CLI (CUPTI-parity). Not
  yet implemented — selecting it warns and falls back to ``torch_events``. See the ``rocm-benchmark``
  skill (``.claude/skills/rocm-benchmark/SKILL.md``).

Historically this module used ``flashinfer.testing.bench_gpu_time_with_cupti``. That path depends on
CUPTI (NVIDIA-only) and, on the ROCm ``flashinfer`` build, exposes an incompatible signature, so it
was replaced with the self-contained ``torch.cuda.Event`` implementation below.
"""

from __future__ import annotations

import os
import statistics
import warnings
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType
from typing import Any, List

import torch

from flashinfer_bench.compile import Runnable

# Device-specific lock registry to ensure multiprocess-safe benchmarking
_device_locks: dict[str, LockType] = {}
_registry_lock = Lock()

# Size (MiB) of the scratch buffer written between timed iterations to evict the last-level cache
# and measure cold-cache latency. Overridable via FIB_L2_FLUSH_MB (0 disables the flush). The
# default (256 MiB) is sized to exceed the L2/Infinity-Cache of current CDNA parts.
_DEFAULT_L2_FLUSH_MB = 256


def _device_lock(device: str) -> LockType:
    """Get or create a multiprocessing lock for the specified device.

    This function maintains a registry of locks per device to ensure that
    benchmarking operations on the same device are serialized, preventing
    interference between concurrent measurements.

    Parameters
    ----------
    device : str
        The device identifier (e.g., "cuda:0", "cuda:1").

    Returns
    -------
    LockType
        A lock object specific to the given device.
    """
    with _registry_lock:
        lock = _device_locks.get(device)
        if lock is None:
            lock = Lock()
            _device_locks[device] = lock
        return lock


def _l2_flush_mb() -> int:
    """Return the cold-cache flush buffer size in MiB (from FIB_L2_FLUSH_MB, default 256)."""
    value = os.environ.get("FIB_L2_FLUSH_MB")
    if value is None:
        return _DEFAULT_L2_FLUSH_MB
    try:
        return max(0, int(value))
    except ValueError:
        return _DEFAULT_L2_FLUSH_MB


def _time_with_torch_events(
    fn: Runnable, args: List[Any], warmup: int, iters: int, device: str
) -> float:
    """Time a value-returning Runnable using ``torch.cuda.Event`` (HIP/CUDA events).

    Performs ``warmup`` untimed iterations, then ``iters`` timed iterations, evicting the
    last-level cache before each timed call (unless disabled), and returns the median latency.

    Parameters
    ----------
    fn : Runnable
        The kernel to benchmark (value-returning style), called as ``fn(*args)``.
    args : List[Any]
        Positional arguments in definition order.
    warmup : int
        Number of untimed warmup iterations.
    iters : int
        Number of timed iterations.
    device : str
        The CUDA/HIP device to run on (e.g. "cuda:0").

    Returns
    -------
    float
        Median execution time in milliseconds.
    """
    targs = tuple(args)

    # Warmup
    for _ in range(max(warmup, 0)):
        fn(*targs)
    torch.cuda.synchronize(device)

    # Cold last-level-cache flush buffer (allocated once, zeroed before each timed iter).
    flush_mb = _l2_flush_mb()
    flush_buf = (
        torch.empty(flush_mb * 1024 * 1024, dtype=torch.int8, device=device)
        if flush_mb > 0
        else None
    )

    n = max(iters, 1)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]

    for i in range(n):
        if flush_buf is not None:
            flush_buf.zero_()
        starts[i].record()
        fn(*targs)
        ends[i].record()

    torch.cuda.synchronize(device)
    times = [starts[i].elapsed_time(ends[i]) for i in range(n)]  # milliseconds
    return statistics.median(times)


def time_runnable(fn: Runnable, args: List[Any], warmup: int, iters: int, device: str) -> float:
    """Time the execution of a value-returning style Runnable kernel.

    Uses ``torch.cuda.Event`` timing (HIP events on ROCm, CUDA events on NVIDIA) by default.
    Benchmarking is serialized per device via a multiprocessing lock.

    Parameters
    ----------
    fn : Runnable
        The kernel function to benchmark (must be value-returning style).
    args : List[Any]
        List of arguments in definition order.
    warmup : int
        Number of warmup iterations before timing.
    iters : int
        Number of timing iterations to average over.
    device : str
        The CUDA/HIP device to run the benchmark on.

    Returns
    -------
    float
        The median execution time in milliseconds.
    """
    backend = os.environ.get("FIB_TIMING_BACKEND", "torch_events").lower()
    if backend == "rocprof":
        # Reserved rocprofv3-CLI backend, not implemented yet: warn and fall back to the portable
        # torch-event timing rather than failing on a documented backend name.
        warnings.warn(
            "FIB_TIMING_BACKEND='rocprof' is not implemented yet; falling back to 'torch_events'.",
            RuntimeWarning,
            stacklevel=2,
        )
        backend = "torch_events"
    if backend not in ("torch_events", ""):
        raise ValueError(
            f"Unsupported FIB_TIMING_BACKEND='{backend}'. Supported: 'torch_events', 'rocprof'."
        )

    lock = _device_lock(device)
    with lock:
        with torch.cuda.device(device):
            return _time_with_torch_events(fn, args, warmup, iters, device)
