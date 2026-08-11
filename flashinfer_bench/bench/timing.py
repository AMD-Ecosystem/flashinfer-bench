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

import logging
import os
import statistics
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockType
from typing import Any, List, Optional, Tuple

import torch

from flashinfer_bench.compile import Runnable

logger = logging.getLogger(__name__)

# Device-specific lock registry to ensure multiprocess-safe benchmarking
_device_locks: dict[str, LockType] = {}
_registry_lock = Lock()

# Size (MiB) of the scratch buffer written between timed iterations to evict the last-level cache
# and measure cold-cache latency. Overridable via FIB_L2_FLUSH_MB (0 disables the flush). The
# default (256 MiB) is sized to exceed the L2/Infinity-Cache of current CDNA parts.
_DEFAULT_L2_FLUSH_MB = 256

# The flush has a second, load-bearing job beyond cold-cache semantics: it keeps the device busy
# while the host enqueues the next iteration. `torch.cuda.Event`s are stream-ordered, so the timed
# window runs from when the device reaches the start marker to when it reaches the end marker. If
# the queue drains first, the device sits idle waiting for the host to launch the kernel and that
# idle time is billed as kernel latency. Measured on gfx942/MI300X: with the 256 MiB default and a
# ~6 us kernel, a solution doing 50/100/200 us of Python per call reported 24/74/175 us -- up to
# 28x its real cost -- while quadrupling the flush brought every case back to ~7 us. Python-wrapped
# solutions (the AITER path) sit squarely in that range, so this is a default-configuration hazard,
# not an exotic one.
#
# So after the primary measurement we re-measure with more cover, repeating until the number stops
# improving. Convergence is the signal that the device is no longer waiting on the host: with 4x
# cover the 50/100/200 us cases above all came back to ~7 us and then held steady, while genuinely
# cheap solutions moved less than 5% (noise) from the first step. If we run out of headroom before
# it converges, the result is reported but marked dispatch-bound rather than presented as fact.
_COVER_ESCALATION_FACTOR = 4
"""Multiplier applied to the cover buffer on each escalation step."""

_COVER_MAX_ESCALATIONS = 3
"""Cap on escalation steps, so a pathological solution cannot spin allocating ever-larger buffers."""

_COVER_MAX_FREE_MEMORY_FRACTION = 0.10
"""Never spend more than this share of free device memory on cover. Deliberately conservative:
benchmark hosts are often shared, and a cover buffer that crowds out other work trades one
measurement artifact for another."""

_COVER_VERIFY_MIN_ITERS = 8
"""Floor on escalation-pass iterations; they only need to resolve a large discrepancy."""

_COVER_IMPROVEMENT_THRESHOLD = 0.10
"""Relative drop below which the measurement is considered converged."""


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


def _alloc_cover_buffer(cover_mb: int, device: str) -> Optional[torch.Tensor]:
    """Allocate the flush/cover buffer, returning None if it does not fit.

    An OOM here is the harness failing to provision itself, not the solution misbehaving, but the
    caller wraps timing in a broad ``except`` that records RUNTIME_ERROR against the solution. So
    degrade to no cover rather than mislabelling someone else's kernel as broken.
    """
    if cover_mb <= 0:
        return None
    try:
        return torch.empty(cover_mb * 1024 * 1024, dtype=torch.int8, device=device)
    except torch.cuda.OutOfMemoryError:
        logger.warning(
            "Could not allocate the %d MiB timing cover buffer on %s; continuing without it. "
            "Measurements may include host dispatch time.",
            cover_mb,
            device,
        )
        return None


def _measure(fn: Runnable, targs: tuple, iters: int, device: str, cover_mb: int) -> float:
    """Run ``iters`` event-timed iterations behind ``cover_mb`` MiB of queued device work.

    Returns the median per-iteration latency in milliseconds.
    """
    cover_buf = _alloc_cover_buffer(cover_mb, device)
    try:
        n = max(iters, 1)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]

        for i in range(n):
            if cover_buf is not None:
                cover_buf.zero_()
            starts[i].record()
            fn(*targs)
            ends[i].record()

        torch.cuda.synchronize(device)
        return statistics.median(starts[i].elapsed_time(ends[i]) for i in range(n))
    finally:
        del cover_buf


def _time_with_torch_events(
    fn: Runnable, args: List[Any], warmup: int, iters: int, device: str
) -> Tuple[float, bool]:
    """Time a value-returning Runnable using ``torch.cuda.Event`` (HIP/CUDA events).

    Performs ``warmup`` untimed iterations, then ``iters`` timed iterations, evicting the
    last-level cache before each timed call (unless disabled). Shorter follow-up passes with
    progressively more cover then check that the device was never left waiting on the host
    mid-window; see the module-level notes on ``_COVER_ESCALATION_FACTOR``. If a pass comes back
    materially faster, the earlier number was measuring Python, and the loop keeps escalating
    until the measurement converges or it runs out of headroom.

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
    Tuple[float, bool]
        Median execution time in milliseconds, and whether the measurement was dispatch-bound
        (i.e. the configured cover was too small, so the reported value may still be inflated).
    """
    targs = tuple(args)

    # Warmup
    for _ in range(max(warmup, 0)):
        fn(*targs)
    torch.cuda.synchronize(device)

    configured_mb = _l2_flush_mb()
    best = _measure(fn, targs, iters, device, configured_mb)

    if configured_mb <= 0:
        # With the flush disabled there is nothing to escalate from: any host dispatch cost lands
        # inside the window by construction. Report the number, but do not claim it is clean.
        return best, True

    escalation_iters = max(_COVER_VERIFY_MIN_ITERS, max(iters, 1) // 4)
    cover_mb = configured_mb

    for _ in range(_COVER_MAX_ESCALATIONS):
        next_mb = cover_mb * _COVER_ESCALATION_FACTOR
        if next_mb > _max_cover_mb(device):
            break
        candidate = _measure(fn, targs, escalation_iters, device, next_mb)
        cover_mb = next_mb
        if candidate >= best * (1.0 - _COVER_IMPROVEMENT_THRESHOLD):
            # More cover stopped helping: the device was never idle inside the window.
            if cover_mb > configured_mb * _COVER_ESCALATION_FACTOR:
                logger.info(
                    "Timing needed %d MiB of cover (FIB_L2_FLUSH_MB=%d was too small for this "
                    "solution's per-call host overhead); converged at %.1f us.",
                    cover_mb // _COVER_ESCALATION_FACTOR,
                    configured_mb,
                    min(best, candidate) * 1000.0,
                )
            return min(best, candidate), False
        best = candidate

    logger.warning(
        "Timing is dispatch-bound: the measurement was still improving at %d MiB of cover "
        "(now %.1f us), so the device is idling inside the timed window while the host enqueues "
        "work. Reporting the lowest value seen; it is an upper bound, not the kernel cost. Reduce "
        "this solution's per-call Python overhead, or raise FIB_L2_FLUSH_MB above %d.",
        cover_mb,
        best * 1000.0,
        configured_mb,
    )
    return best, True


def _max_cover_mb(device: str) -> float:
    """Largest cover buffer worth allocating: a slice of free device memory, not all of it."""
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        # No reliable reading (e.g. a stubbed device in tests): fall back to the configured size
        # only, i.e. do not escalate.
        return 0.0
    return free_bytes * _COVER_MAX_FREE_MEMORY_FRACTION / (1024 * 1024)


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
    latency_ms, _ = time_runnable_detailed(fn, args, warmup, iters, device)
    return latency_ms


def time_runnable_detailed(
    fn: Runnable, args: List[Any], warmup: int, iters: int, device: str
) -> Tuple[float, bool]:
    """Time a Runnable and report whether the measurement was dispatch-bound.

    Same measurement as :func:`time_runnable`, but also returns the confidence flag so callers
    that record a trace can mark a latency the harness knows may be inflated. See
    :func:`_time_with_torch_events`.

    Returns
    -------
    Tuple[float, bool]
        Median execution time in milliseconds, and True when the device was left waiting on the
        host inside the timed window (so the latency is an upper bound, not the kernel cost).
    """
    backend = os.environ.get("FIB_TIMING_BACKEND", "torch_events").lower()
    if backend == "rocprof":
        # Reserved rocprofv3-CLI backend, not implemented yet: fall back to the portable
        # torch-event timing rather than failing on a documented backend name.
        #
        # Logged, not warnings.warn: the default warning filter shows a given warning once per
        # process, so over a long run every trace after the first was silently wall-clock timing
        # while the operator believed they had asked for device-side kernel timing.
        logger.warning(
            "FIB_TIMING_BACKEND='rocprof' is not implemented yet; measuring with 'torch_events' "
            "instead. These are wall-clock event timings, not rocprofv3 kernel durations."
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
