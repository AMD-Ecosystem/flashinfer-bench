"""CPU-safe unit tests for the rocprofv3 agent profiler wiring and error handling."""

from __future__ import annotations

from flashinfer_bench.agents import flashinfer_bench_run_rocprof, get_all_tool_schemas
from flashinfer_bench.agents.rocprof import _format_kernel_report, _read_csv

_KERNEL_HEADER = "Kernel_Name,Start_Timestamp,End_Timestamp\n"


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


def test_read_csv_merges_all_matches_in_sorted_order(tmp_path):
    """rocprofv3 emits one CSV per process; all matches are read, deterministically ordered."""
    (tmp_path / "b_kernel_trace.csv").write_text(_KERNEL_HEADER + "kb,100,200\n")
    (tmp_path / "a_kernel_trace.csv").write_text(_KERNEL_HEADER + "ka,10,20\n")

    rows = _read_csv(str(tmp_path / "**" / "*kernel_trace*.csv"))

    assert [r["Kernel_Name"] for r in rows] == ["ka", "kb"]


def test_format_kernel_report_skips_unparsable_timestamps(tmp_path):
    """Non-integer timestamps are dropped, not crashed on, in the region-fallback path.

    The marker region window (1000-1100) excludes every kernel, so region correlation selects
    nothing and the fallback over all rows runs — that is the path that used to call ``int()``
    unguarded on a row whose timestamp is merely truthy.
    """
    (tmp_path / "marker_api_trace.csv").write_text(
        "Name,Start_Timestamp,End_Timestamp\nflashinfer_bench_ncu_profile,1000,1100\n"
    )
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER + "bad_kernel,not_a_number,also_bad\ngood_kernel,5000,6000\n"
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "good_kernel" in out
    assert "bad_kernel" not in out
    assert "1 kernel dispatch(es)" in out


def test_format_kernel_report_no_output(tmp_path):
    assert _format_kernel_report(tmp_path, max_lines=None).startswith("ERROR:")
