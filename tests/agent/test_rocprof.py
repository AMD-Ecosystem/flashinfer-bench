"""CPU-safe unit tests for the rocprofv3 agent profiler wiring and error handling."""

from __future__ import annotations

from flashinfer_bench.agents import flashinfer_bench_run_rocprof, get_all_tool_schemas
from flashinfer_bench.agents._profiling import PROFILE_REGION
from flashinfer_bench.agents.rocprof import _format_kernel_report, _read_csv

_KERNEL_HEADER = "Kernel_Name,Start_Timestamp,End_Timestamp\n"

# Real rocprofv3 marker_api_trace.csv header, captured from ROCm 7.2 in the docker/rocm container.
# The roctx message lands in "Function" — there is NO "Name" column, which is what made region
# correlation silently no-op. Fixtures must use the real schema or they re-encode that bug.
_MARKER_HEADER = (
    "Domain,Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp\n"
)


def _marker_row(name: str, start, end, domain: str = "MARKER_CORE_RANGE_API") -> str:
    """One marker_api_trace row in real rocprofv3 layout."""
    return f"{domain},{name},11,11,2,{start},{end}\n"


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
        _MARKER_HEADER + _marker_row(PROFILE_REGION, 1000, 1100)
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


def test_region_window_skips_malformed_marker_rows(tmp_path):
    """A malformed marker row must not disable region scoping when a valid one follows.

    Merging every marker CSV makes multiple region rows likely on multi-process runs, so
    bailing out on the first unparsable one would silently widen the report to all kernels.
    """
    (tmp_path / "marker_api_trace.csv").write_text(
        _MARKER_HEADER
        + _marker_row(PROFILE_REGION, "bogus", "bogus")
        + _marker_row(PROFILE_REGION, 1000, 9000)
    )
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER + "in_window,2000,3000\nout_of_window,20000,30000\n"
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "in_window" in out
    assert "out_of_window" not in out
    assert "1 kernel dispatch(es)" in out


def test_region_scoping_works_against_real_rocprofv3_schema(tmp_path):
    """The regression test for the bug this file's fixtures used to hide.

    Captured verbatim from ROCm 7.2 in the docker/rocm container: the roctx message is in the
    "Function" column and there is no "Name" column at all. Reading "Name" made _region_window
    return None on every real run, so the report silently widened to every kernel on the device
    while still advertising the region.
    """
    (tmp_path / "marker_api_trace.csv").write_text(
        _MARKER_HEADER
        + f"MARKER_CORE_RANGE_API,{PROFILE_REGION},11,11,2,882284644452133,882284801383134\n"
    )
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER
        + "in_region,882284650000000,882284660000000\n"
        + "warmup_before_region,882284000000000,882284100000000\n"
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "in_region" in out
    assert "warmup_before_region" not in out
    assert "1 kernel dispatch(es)" in out
    assert f"region '{PROFILE_REGION}'" in out
    assert "ALL kernels" not in out


def test_region_scoping_accepts_legacy_name_column(tmp_path):
    """ROCm 6.4.1 and 7.2 both use "Function"; "Name" is tolerated in case a build differs."""
    (tmp_path / "marker_api_trace.csv").write_text(
        f"Name,Start_Timestamp,End_Timestamp\n{PROFILE_REGION},1000,9000\n"
    )
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER + "in_window,2000,3000\nout_of_window,20000,30000\n"
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "in_window" in out
    assert "out_of_window" not in out


def test_header_admits_when_scoping_failed(tmp_path):
    """An unscoped report must not claim to be scoped — that is what misdirects an agent."""
    # No marker CSV at all -> correlation cannot happen.
    (tmp_path / "kernel_trace.csv").write_text(_KERNEL_HEADER + "k1,1000,2000\nk2,3000,4000\n")

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "ALL kernels" in out
    assert "not found in the marker trace" in out
    assert "2 kernel dispatch(es)" in out
    # It must not read as a scoped report.
    assert f"(region '{PROFILE_REGION}'," not in out


def test_launch_geometry_labelled_as_work_items_not_grid(tmp_path):
    """Grid_Size_* is work-items (HSA), not CUDA gridDim; the label must not imply workgroups."""
    (tmp_path / "kernel_trace.csv").write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp,Grid_Size_X,Grid_Size_Y,Grid_Size_Z,"
        "Workgroup_Size_X,Workgroup_Size_Y,Workgroup_Size_Z\n"
        "k,1000,2000,262144,1,1,256,1,1\n"
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "items 262144x1x1" in out
    assert "wg 256x1x1" in out
    assert "grid 262144" not in out  # would read as gridDim and overstate the launch 256x


def test_format_kernel_report_errors_when_no_row_parses(tmp_path):
    """Rows present but none parsable is a failure, not a legitimate zero-kernel run."""
    (tmp_path / "kernel_trace.csv").write_text(_KERNEL_HEADER + "k1,nope,nope\nk2,,\n")

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert out.startswith("ERROR:")
    assert "2 row(s)" in out
    assert "0 kernel dispatch(es)" not in out
