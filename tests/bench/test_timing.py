"""Tests for bench/timing.py.

This module had no coverage at all, which is how the dispatch-bound bug below survived: the
evaluator tests monkeypatch ``time_runnable`` to a constant, so nothing exercised the real
measurement path.

The GPU tests here are deliberately built around a *known-cheap kernel with expensive Python*,
because that is the shape that produces a wrong answer -- see the module docstring in
``flashinfer_bench.bench.timing``.
"""

import logging
import time

import pytest
import torch

from flashinfer_bench.bench import timing
from flashinfer_bench.bench.timing import _l2_flush_mb, time_runnable, time_runnable_detailed

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device not available"
)


# --------------------------------------------------------------------------------------------
# Environment parsing (CPU-only)
# --------------------------------------------------------------------------------------------


def test_l2_flush_mb_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("FIB_L2_FLUSH_MB", raising=False)
    assert _l2_flush_mb() == 256


@pytest.mark.parametrize("value,expected", [("0", 0), ("64", 64), ("-8", 0)])
def test_l2_flush_mb_parses_and_clamps(monkeypatch, value, expected):
    """Negative sizes clamp to 0 (disabled) rather than blowing up in torch.empty."""
    monkeypatch.setenv("FIB_L2_FLUSH_MB", value)
    assert _l2_flush_mb() == expected


def test_l2_flush_mb_falls_back_on_garbage(monkeypatch):
    """A typo'd value silently reverts to the default -- pin the behaviour so it stays deliberate."""
    monkeypatch.setenv("FIB_L2_FLUSH_MB", "not-a-number")
    assert _l2_flush_mb() == 256


def test_unsupported_backend_raises(monkeypatch):
    monkeypatch.setenv("FIB_TIMING_BACKEND", "cupti")
    with pytest.raises(ValueError, match="Unsupported FIB_TIMING_BACKEND"):
        time_runnable(lambda: None, [], 0, 1, "cuda:0")


def test_rocprof_backend_warns_on_every_call(monkeypatch, caplog):
    """The fallback must be loud each time, not once per process.

    ``warnings.warn`` is deduplicated by the default filter, so in a long run only the first trace
    carried the notice and every later one silently reported wall-clock timings while the operator
    believed rocprofv3 kernel durations had been requested.
    """
    monkeypatch.setenv("FIB_TIMING_BACKEND", "rocprof")
    monkeypatch.setattr(timing, "_time_with_torch_events", lambda *a, **k: (1.0, False))

    with caplog.at_level(logging.WARNING, logger="flashinfer_bench.bench.timing"):
        for _ in range(3):
            time_runnable(lambda: None, [], 0, 1, "cuda:0")

    assert sum("not implemented yet" in r.message for r in caplog.records) == 3


# --------------------------------------------------------------------------------------------
# Dispatch-bound detection (GPU)
# --------------------------------------------------------------------------------------------


def _busy_wait(microseconds: float) -> None:
    """Burn host CPU without touching the GPU."""
    deadline = time.perf_counter() + microseconds * 1e-6
    while time.perf_counter() < deadline:
        pass


def _make_runnable(python_overhead_us: float, tensor: torch.Tensor, out: torch.Tensor):
    """A trivial kernel fronted by a configurable amount of per-call Python."""

    def run(x):
        if python_overhead_us:
            _busy_wait(python_overhead_us)
        torch.add(x, 1.0, out=out)
        return out

    return run


@requires_cuda
def test_cheap_kernel_is_not_flagged(monkeypatch):
    """A solution with negligible Python overhead must not be reported as dispatch-bound."""
    monkeypatch.delenv("FIB_L2_FLUSH_MB", raising=False)
    device = "cuda:0"
    x = torch.randn(4096, device=device)
    out = torch.empty_like(x)

    latency_ms, dispatch_bound = time_runnable_detailed(
        _make_runnable(0, x, out), [x], warmup=20, iters=100, device=device
    )

    assert not dispatch_bound
    assert latency_ms > 0.0


@requires_cuda
def test_python_heavy_solution_is_corrected_not_inflated(monkeypatch):
    """The regression guard.

    Same kernel in both measurements; the second is fronted by ~200 us of per-call Python. Before
    this fix the second reported ~175 us -- roughly 28x the kernel's real cost -- with nothing to
    indicate the number was host time rather than device time. Escalating the cover must bring it
    back to the kernel's actual cost.
    """
    monkeypatch.delenv("FIB_L2_FLUSH_MB", raising=False)
    device = "cuda:0"
    x = torch.randn(4096, device=device)
    out = torch.empty_like(x)

    baseline_ms, _ = time_runnable_detailed(
        _make_runnable(0, x, out), [x], warmup=20, iters=100, device=device
    )
    heavy_ms, _ = time_runnable_detailed(
        _make_runnable(200, x, out), [x], warmup=20, iters=100, device=device
    )

    # The kernel is identical, so the reported latency must track the kernel, not the Python.
    assert heavy_ms < baseline_ms * 5, (
        f"latency {heavy_ms * 1000:.1f} us should be close to the kernel cost "
        f"{baseline_ms * 1000:.1f} us, not the ~200 us of per-call Python"
    )


@requires_cuda
def test_uncorrectable_dispatch_bound_is_flagged(monkeypatch):
    """When escalation cannot fix it, the number ships flagged rather than presented as fact.

    Forced by denying the escalation headroom, which is what happens on a busy device.
    """
    monkeypatch.delenv("FIB_L2_FLUSH_MB", raising=False)
    monkeypatch.setattr(timing, "_max_cover_mb", lambda device: 0.0)
    device = "cuda:0"
    x = torch.randn(4096, device=device)
    out = torch.empty_like(x)

    heavy_ms, dispatch_bound = time_runnable_detailed(
        _make_runnable(200, x, out), [x], warmup=20, iters=60, device=device
    )

    assert dispatch_bound, "no headroom to verify -> must not be certified clean"
    assert heavy_ms > 0.0


@requires_cuda
def test_disabled_flush_is_reported_as_dispatch_bound(monkeypatch):
    """With no cover there is nothing to escalate from, so the result cannot be certified clean."""
    monkeypatch.setenv("FIB_L2_FLUSH_MB", "0")
    device = "cuda:0"
    x = torch.randn(4096, device=device)
    out = torch.empty_like(x)

    _, dispatch_bound = time_runnable_detailed(
        _make_runnable(0, x, out), [x], warmup=10, iters=50, device=device
    )

    assert dispatch_bound


@requires_cuda
def test_time_runnable_returns_bare_float(monkeypatch):
    """The original API must keep its signature; only the detailed variant exposes the flag."""
    monkeypatch.delenv("FIB_L2_FLUSH_MB", raising=False)
    device = "cuda:0"
    x = torch.randn(1024, device=device)
    out = torch.empty_like(x)

    result = time_runnable(_make_runnable(0, x, out), [x], warmup=5, iters=20, device=device)

    assert isinstance(result, float)
