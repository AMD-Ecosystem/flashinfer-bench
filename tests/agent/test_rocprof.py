"""CPU-safe unit tests for the rocprofv3 agent profiler wiring and error handling."""

from __future__ import annotations

from flashinfer_bench.agents import flashinfer_bench_run_rocprof, get_all_tool_schemas
from flashinfer_bench.agents._profiling import PROFILE_REGION, UNSCOPED_MARKER
from flashinfer_bench.agents.rocprof import _format_kernel_report, _read_csv

# Real rocprofv3 CSV headers, captured from ROCm 7.2 in the docker/rocm container.
#
# Fixtures must use the real layout or they re-encode the bug they are meant to catch: the marker
# trace has NO "Name" column (the roctx message is in "Function"), which is what let region
# correlation silently no-op while the tests stayed green. The kernel trace likewise carries far
# more columns than the code reads — writing a minimal invented header hides column-name drift the
# same way.
_MARKER_HEADER = (
    "Domain,Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp\n"
)
_KERNEL_HEADER = (
    "Kind,Agent_Id,Queue_Id,Stream_Id,Thread_Id,Dispatch_Id,Kernel_Id,Kernel_Name,Correlation_Id,"
    "Start_Timestamp,End_Timestamp,LDS_Block_Size,Scratch_Size,VGPR_Count,Accum_VGPR_Count,"
    "SGPR_Count,Workgroup_Size_X,Workgroup_Size_Y,Workgroup_Size_Z,"
    "Grid_Size_X,Grid_Size_Y,Grid_Size_Z\n"
)


def _marker_row(name: str, start, end, domain: str = "MARKER_CORE_RANGE_API") -> str:
    """One marker_api_trace row in real rocprofv3 layout."""
    return f"{domain},{name},11,11,2,{start},{end}\n"


def _kernel_row(name: str, start, end, grid_x=262144, wg_x=256) -> str:
    """One kernel_trace row in real rocprofv3 (ROCm 7.2) layout."""
    return (
        f"KERNEL_DISPATCH,2,1,0,11,1,8,{name},1,{start},{end},"
        f"18432,0,32,0,80,{wg_x},1,1,{grid_x},1,1\n"
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


def test_read_csv_merges_all_matches_in_sorted_order(tmp_path):
    """rocprofv3 emits one CSV per process; all matches are read, deterministically ordered."""
    (tmp_path / "b_kernel_trace.csv").write_text(_KERNEL_HEADER + _kernel_row("kb", 100, 200))
    (tmp_path / "a_kernel_trace.csv").write_text(_KERNEL_HEADER + _kernel_row("ka", 10, 20))

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
        _KERNEL_HEADER
        + _kernel_row("bad_kernel", "not_a_number", "also_bad")
        + _kernel_row("good_kernel", 5000, 6000)
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
        _KERNEL_HEADER
        + _kernel_row("in_window", 2000, 3000)
        + _kernel_row("out_of_window", 20000, 30000)
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
        _MARKER_HEADER + _marker_row(PROFILE_REGION, 882284644452133, 882284801383134)
    )
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER
        + _kernel_row("in_region", 882284650000000, 882284660000000)
        + _kernel_row("warmup_before_region", 882284000000000, 882284100000000)
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "in_region" in out
    assert "warmup_before_region" not in out
    assert "1 kernel dispatch(es)" in out
    assert f"region '{PROFILE_REGION}'" in out
    assert UNSCOPED_MARKER not in out
    # The count must prove filtering happened, not just that a region was named.
    assert "(1 of 2)" in out


def test_region_scoping_accepts_legacy_name_column(tmp_path):
    """ROCm 6.4.1 and 7.2 both use "Function"; "Name" is tolerated in case a build differs."""
    (tmp_path / "marker_api_trace.csv").write_text(
        f"Name,Start_Timestamp,End_Timestamp\n{PROFILE_REGION},1000,9000\n"
    )
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER
        + _kernel_row("in_window", 2000, 3000)
        + _kernel_row("out_of_window", 20000, 30000)
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "in_window" in out
    assert "out_of_window" not in out


def test_region_found_when_another_column_is_populated_but_unrelated(tmp_path):
    """Every candidate column is searched, not just the first populated one.

    A build that fills "Function" with the API name and puts the roctx message in "Name" would
    defeat a first-truthy-wins lookup and silently lose scoping — the exact failure this fix
    exists to remove, reintroduced one layer down.
    """
    (tmp_path / "marker_api_trace.csv").write_text(
        "Domain,Function,Name,Start_Timestamp,End_Timestamp\n"
        f"MARKER_CORE_RANGE_API,roctxRangePushA,{PROFILE_REGION},1000,9000\n"
    )
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER
        + _kernel_row("in_window", 2000, 3000)
        + _kernel_row("out_of_window", 20000, 30000)
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "in_window" in out
    assert "out_of_window" not in out
    assert UNSCOPED_MARKER not in out


def test_header_admits_when_scoping_failed(tmp_path):
    """An unscoped report must not claim to be scoped — that is what misdirects an agent."""
    # No marker CSV at all -> correlation cannot happen.
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER + _kernel_row("k1", 1000, 2000) + _kernel_row("k2", 3000, 4000)
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert UNSCOPED_MARKER in out
    assert "not found in the marker trace" in out
    assert "2 kernel dispatch(es)" in out
    # It must not read as a scoped report.
    assert f"(region '{PROFILE_REGION}'," not in out


def test_disjoint_spans_do_not_merge_into_one_window(tmp_path):
    """Two per-process regions must stay disjoint, not collapse into their bounding box.

    _read_csv merges every marker CSV, so a multi-process run yields one span per process.
    Taking (min start, max end) would bridge the gap and readmit whatever ran between them —
    silently unscoped again, while the header still claims success.
    """
    (tmp_path / "marker_api_trace.csv").write_text(
        _MARKER_HEADER
        + _marker_row(PROFILE_REGION, 1000, 2000)
        + _marker_row(PROFILE_REGION, 8000, 9000)
    )
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER
        + _kernel_row("in_first_span", 1100, 1200)
        + _kernel_row("between_spans", 4000, 5000)  # inside the bounding box, inside no span
        + _kernel_row("in_second_span", 8100, 8200)
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "in_first_span" in out
    assert "in_second_span" in out
    assert "between_spans" not in out
    assert "(2 of 3)" in out


def test_header_admits_when_window_matched_no_dispatch(tmp_path):
    """The other scoping-failure branch: a window was found but excludes every kernel.

    Reachable on real hardware — a kernel's End_Timestamp can land after the host-side roctx pop.
    Without this the branch is untested and the scope/selection pairing has nothing guarding it.
    """
    (tmp_path / "marker_api_trace.csv").write_text(
        _MARKER_HEADER + _marker_row(PROFILE_REGION, 1000, 1100)
    )
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER + _kernel_row("way_after", 900000, 910000)
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert UNSCOPED_MARKER in out
    assert "matched no dispatch" in out
    assert "way_after" in out  # fell back to all kernels
    assert f"(region '{PROFILE_REGION}'," not in out


def test_launch_geometry_labelled_as_work_items_not_grid(tmp_path):
    """Grid_Size_* is work-items (HSA), not CUDA gridDim; the label must not imply workgroups."""
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER + _kernel_row("k", 1000, 2000, grid_x=262144, wg_x=256)
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert "items 262144x1x1" in out
    assert "wg 256x1x1" in out
    assert "grid 262144" not in out  # would read as gridDim and overstate the launch 256x


def test_short_rows_do_not_raise(tmp_path):
    """csv.DictReader pads short rows with None; formatting None must not escape as a traceback.

    The module contract is that failures come back as "ERROR:" strings — an unhandled
    TypeError from f"{None:>4}" breaks it for every caller.
    """
    (tmp_path / "kernel_trace.csv").write_text(
        "Kernel_Name,Start_Timestamp,End_Timestamp,VGPR_Count,Grid_Size_X\nk1,1000,2000\n"
    )

    out = _format_kernel_report(tmp_path, max_lines=None)  # must not raise

    assert "k1" in out
    assert "VGPR    ?" in out  # missing field renders as the placeholder, not None


def test_format_kernel_report_errors_when_no_row_parses(tmp_path):
    """Rows present but none parsable is a failure, not a legitimate zero-kernel run."""
    (tmp_path / "kernel_trace.csv").write_text(
        _KERNEL_HEADER + _kernel_row("k1", "nope", "nope") + _kernel_row("k2", "", "")
    )

    out = _format_kernel_report(tmp_path, max_lines=None)

    assert out.startswith("ERROR:")
    assert "2 row(s)" in out
    assert "0 kernel dispatch(es)" not in out
