"""CPU-safe unit tests for the rocprofv3 agent profiler wiring and error handling."""

from __future__ import annotations

from flashinfer_bench.agents import (
    flashinfer_bench_run_rocprof,
    get_all_tool_schemas,
)


def test_rocprof_in_tool_schema():
    names = {s["name"] for s in get_all_tool_schemas()}
    assert "flashinfer_bench_run_rocprof" in names
    assert "flashinfer_bench_list_rocprof_options" in names
    # NCU is replaced by rocprof in the agent-facing toolset on ROCm.
    assert "flashinfer_bench_run_ncu" not in names


def test_run_rocprof_missing_solution_file():
    out = flashinfer_bench_run_rocprof("/no/such/solution.json", "/no/such/workload.json")
    assert out.startswith("ERROR:")
    assert "Solution file not found" in out


def test_run_rocprof_missing_workload_file(tmp_path):
    sol = tmp_path / "sol.json"
    sol.write_text('{"not": "valid"}')  # parse error path
    out = flashinfer_bench_run_rocprof(str(sol), "/no/such/workload.json")
    assert out.startswith("ERROR:")
