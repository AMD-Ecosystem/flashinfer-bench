"""The marker region shared between the solution runner and the profiling tools.

``_solution_runner`` emits this range around the profiled (non-warmup) run; ``rocprof`` and ``ncu``
correlate against it to scope their reports. Emitter and readers must agree exactly — if they
drift, region correlation silently finds nothing and the report quietly widens to every kernel on
the device rather than failing. Hence one constant instead of a literal per call site.
"""

from __future__ import annotations

__all__ = ["PROFILE_REGION"]

#: Tool-agnostic: read by the rocprofv3 (roctx) tool as well as the NCU (NVTX) one.
PROFILE_REGION = "flashinfer_bench_profile"
