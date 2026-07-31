"""rocprofv3-based profiling tool for LLM agents (ROCm replacement for NCU).

Runs a solution under ``rocprofv3`` and reports per-kernel device-side timing plus launch geometry
(work-items and workgroup size), scoped to the profiled region marked by the runner's roctx range.
Optionally collects hardware performance counters via ``--pmc``.

Occupancy resources (VGPR/SGPR/LDS/scratch) are reported when the profiler emits them: ROCm 7.x
kernel traces carry those columns, ROCm 6.4.x does not (it has Private_Segment_Size /
Group_Segment_Size and no register counts), so on older ROCm those fields read "?" and the report
says so once rather than per line.

All inputs/outputs are JSON-serializable, so this is usable as an LLM agent tool (mirrors the
contract of the former NCU tool). Errors are returned as strings starting with "ERROR:".
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from flashinfer_bench.data import Solution, TraceSet, Workload

from ._output import truncate
from ._profiling import PROFILE_REGION, UNSCOPED_MARKER

logger = logging.getLogger(__name__)


def flashinfer_bench_list_rocprof_options(rocprof_avail_path: str = "rocprofv3-avail") -> str:
    """List available hardware performance counters (PMC) for ``--pmc`` on this GPU.

    Parameters
    ----------
    rocprof_avail_path : str
        Path to the ``rocprofv3-avail`` executable. Default "rocprofv3-avail".

    Returns
    -------
    str
        The available counters, or an error string starting with "ERROR:".
    """
    if shutil.which(rocprof_avail_path) is None:
        return f"ERROR: '{rocprof_avail_path}' not found. Install ROCm rocprofiler-sdk."
    for args in (["list", "pmc"], ["--list-metrics"], ["list"]):
        try:
            r = subprocess.run(
                [rocprof_avail_path, *args], capture_output=True, text=True, timeout=30
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if r.returncode == 0 and (r.stdout or r.stderr):
            return (r.stdout or "") + (r.stderr or "")
    return "ERROR: could not list rocprof PMC options."


def _build_cmd(
    data_dir: Path,
    out_dir: Path,
    device: str,
    trace_set_path: Optional[Path],
    counters: Optional[List[str]],
    rocprof_path: str,
) -> List[str]:
    """Build the rocprofv3 command line wrapping the solution runner."""
    cmd = [
        rocprof_path,
        "--marker-trace",  # capture the roctx region for scoping
        "--kernel-trace",  # per-kernel device timing + resource usage
        "--output-format",
        "csv",
        "-d",
        str(out_dir),
    ]
    if counters:
        cmd += ["--pmc", *counters]
    runner = [
        sys.executable,
        "-u",
        "-m",
        "flashinfer_bench.agents._solution_runner",
        "--data-dir",
        str(data_dir),
        "--device",
        device,
    ]
    if trace_set_path:
        runner += ["--trace-set-path", str(trace_set_path)]
    return cmd + ["--", *runner]


def _read_csv(pattern: str) -> List[dict]:
    """Read every CSV matching a glob into one list of dict rows (empty if none).

    rocprofv3 emits one CSV per profiled process, so a run can produce several matches. Read all
    of them in sorted order instead of an arbitrary glob hit, which would otherwise parse a
    nondeterministic (and possibly wrong) file.
    """
    rows: List[dict] = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


# Column holding the roctx message in marker_api_trace.csv. rocprofv3 writes it to "Function"
# (verified on ROCm 6.4.1 and 7.2); "Name" is accepted as a fallback in case a build differs.
# Do NOT filter on the "Domain" column: it is "MARKER_CORE_API" on 6.4.1 but
# "MARKER_CORE_RANGE_API" on 7.2, so matching it would re-break scoping on one of the two.
_MARKER_NAME_COLUMNS = ("Function", "Name")


def _region_spans(out_dir: Path) -> List[tuple]:
    """Return every (start_ns, end_ns) span carrying the profiled roctx region, in no order.

    A list rather than one window on purpose. ``_read_csv`` merges every marker CSV, so a
    multi-process run yields one span per process. Collapsing them to (min start, max end) would
    bridge the gap between disjoint ranges and silently readmit the kernels that ran *between*
    them — the same "scoped but actually unscoped" failure this module exists to prevent, just
    with extra steps. Callers test membership against any span instead.

    Order is deliberately unspecified: both consumers are order-independent (``any(...)`` over the
    spans, and an emptiness check), so sorting would be work no caller asks for.
    """
    spans = []
    for r in _read_csv(str(out_dir / "**" / "*marker_api_trace*.csv")):
        # Search every candidate column, rather than taking the first populated one: a build
        # that emits both (say "Function" holding the API name "roctxRangePushA" and the message
        # in "Name") would otherwise match the wrong column and silently lose scoping again.
        if not any(PROFILE_REGION in (r.get(c) or "") for c in _MARKER_NAME_COLUMNS):
            continue
        try:
            spans.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"])))
        except (KeyError, TypeError, ValueError):
            # Multi-process runs emit several marker CSVs (all merged by _read_csv), so a
            # malformed row is not the last word — keep scanning for a usable one rather
            # than silently giving up on region scoping.
            continue

    return spans


def _format_kernel_report(out_dir: Path, max_lines: Optional[int]) -> str:
    """Summarize kernel-trace rows within the profiled region."""
    rows = _read_csv(str(out_dir / "**" / "*kernel_trace*.csv"))
    if not rows:
        return "ERROR: no kernel-trace output produced by rocprofv3."

    # Parse timestamps once, up front: rows rocprofv3 emits with missing or non-integer
    # timestamps are dropped here, so neither the windowed nor the fallback path can raise.
    parsed = []
    for r in rows:
        try:
            start, end = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        parsed.append((r, start, end))

    # Rows present but none parsable means the CSV schema or the run is broken. Report that as an
    # error rather than a "0 kernel dispatch(es)" report, which reads like a legitimate empty run.
    if not parsed:
        return (
            f"ERROR: kernel-trace CSV had {len(rows)} row(s) but none carried parsable "
            f"Start_Timestamp/End_Timestamp values; cannot build a profile."
        )

    spans = _region_spans(out_dir)
    all_kernels = [(r, end - start) for r, start, end in parsed]
    # In-region means inside ANY span, not inside their bounding box — disjoint per-process ranges
    # must not merge into one window that swallows whatever ran between them.
    in_region = [
        (r, end - start)
        for r, start, end in parsed
        if any(s <= start and end <= e for s, e in spans)
    ]

    # Assign scope and selection together. Scoping can fail two ways — no marker row matched, or
    # the window matched no dispatch — and both fall back to every kernel in the trace, which then
    # includes the runner's warmup dispatch and any JIT/setup work. An unscoped report that claims
    # to be scoped sends agents optimizing kernels the solution never ran, so the header has to say
    # which happened. Deriving both in one branch keeps them from drifting apart: computing the
    # label and then mutating the selection underneath it is how the header starts lying again.
    if in_region:
        selected, scope = (
            in_region,
            f"region '{PROFILE_REGION}' ({len(in_region)} of {len(all_kernels)})",
        )
    else:
        why = "not found in the marker trace" if not spans else "matched no dispatch"
        selected = all_kernels
        scope = f"{UNSCOPED_MARKER} — region '{PROFILE_REGION}' {why}"

    selected.sort(key=lambda x: x[1], reverse=True)
    lines = [
        f"rocprofv3 kernel profile ({scope}, "
        f"{len(selected)} kernel dispatch(es), sorted by duration):",
        "",
    ]

    # csv.DictReader fills missing trailing fields with None (short/partially-flushed rows, or a
    # merged per-process CSV with a different column count), and f"{None:>4}" raises TypeError —
    # which would escape as a traceback and break the "errors come back as ERROR: strings"
    # contract. Coerce through this rather than trusting .get()'s default.
    def field(row: dict, key: str) -> str:
        value = row.get(key)
        return "?" if value is None or value == "" else str(value)

    for r, dur_ns in selected:
        name = (r.get("Kernel_Name") or "?").strip('"')
        if len(name) > 80:
            name = name[:77] + "..."
        lines.append(
            f"  {dur_ns / 1000.0:9.3f} us | VGPR {field(r, 'VGPR_Count'):>4} "
            f"SGPR {field(r, 'SGPR_Count'):>4} LDS {field(r, 'LDS_Block_Size'):>6} "
            f"scratch {field(r, 'Scratch_Size'):>6} | "
            # Grid_Size_* is a work-item count (HSA semantics), NOT CUDA gridDim. Labelling it
            # "grid" beside "block" would read as workgroups and overstate the launch by the
            # workgroup size, so say "items" explicitly.
            f"items {field(r, 'Grid_Size_X')}x{field(r, 'Grid_Size_Y')}x{field(r, 'Grid_Size_Z')} "
            f"wg {field(r, 'Workgroup_Size_X')}x{field(r, 'Workgroup_Size_Y')}"
            f"x{field(r, 'Workgroup_Size_Z')}"
        )
        lines.append(f"    {name}")

    # Say once why the occupancy fields are blank, instead of leaving N lines of bare "?" that read
    # as "this kernel has no registers" rather than "this profiler build does not report them".
    # Gate on a real value, not on key presence: DictReader creates the key with None for a short
    # row, so a header that merely declares the column would suppress this note while every line
    # still printed "?". Reuse field() so the test matches exactly what was rendered. Scan every
    # selected row, not just the first — _read_csv merges per-process CSVs that can carry
    # different headers, so keying off one row decides by luck of sort order.
    if selected and not any(
        field(r, k) != "?" for r, _ in selected for k in ("VGPR_Count", "LDS_Block_Size")
    ):
        lines += [
            "",
            "NOTE: this rocprofv3 build's kernel trace carries no occupancy columns "
            "(ROCm 6.4.x exposes Private_Segment_Size/Group_Segment_Size and no register counts), "
            "so VGPR/SGPR/LDS/scratch read '?'. ROCm 7.x reports them.",
        ]

    # Note whether counters (PMC) were collected. Only the row count is reported — a full counter
    # dump would dwarf the kernel report, so the values stay in the CSV on disk.
    pmc = _read_csv(str(out_dir / "**" / "*counter_collection*.csv"))
    if pmc:
        lines += ["", f"Hardware counters ({len(pmc)} rows) — see counter_collection CSV."]

    return truncate("\n".join(lines), max_lines)


def flashinfer_bench_run_rocprof(
    solution: Union[Solution, str],
    workload: Union[Workload, str],
    *,
    device: str = "cuda:0",
    trace_set_path: Optional[str] = None,
    counters: Optional[List[str]] = None,
    rocprof_path: str = "rocprofv3",
    timeout: int = 120,
    tmpdir: Optional[str] = None,
    max_lines: Optional[int] = None,
) -> str:
    """Profile a solution+workload with rocprofv3 and return a kernel report.

    Parameters
    ----------
    solution : Solution or str
        Solution object or path to a solution JSON file.
    workload : Workload or str
        Workload object or path to a workload JSON file.
    device : str
        Device to run on (ROCm uses the "cuda" device string). Default "cuda:0".
    trace_set_path : str, optional
        Path to the trace set. Defaults to the FIB_DATASET_PATH environment variable.
    counters : List[str], optional
        Hardware performance counters to collect via ``--pmc`` (see
        ``flashinfer_bench_list_rocprof_options``).
    rocprof_path : str
        Path to the ``rocprofv3`` executable. Default "rocprofv3".
    timeout : int
        Timeout in seconds. Default 120.
    tmpdir : str, optional
        Temporary directory for build/profile artifacts.
    max_lines : int, optional
        Truncate output to this many lines.

    Returns
    -------
    str
        The kernel profile report, or an error string starting with "ERROR:".
    """
    if isinstance(solution, str):
        p = Path(solution)
        if not p.exists():
            return f"ERROR: Solution file not found: {solution}"
        try:
            solution = Solution.model_validate_json(p.read_text())
        except Exception as e:
            return f"ERROR: Failed to parse solution file: {e}"

    if isinstance(workload, str):
        p = Path(workload)
        if not p.exists():
            return f"ERROR: Workload file not found: {workload}"
        try:
            workload = Workload.model_validate_json(p.read_text())
        except Exception as e:
            return f"ERROR: Failed to parse workload file: {e}"

    try:
        trace_set = TraceSet.from_path(trace_set_path)
    except Exception as e:
        return f"ERROR: Failed to load trace set: {e}"

    if solution.definition not in trace_set.definitions:
        return (
            f"ERROR: Definition '{solution.definition}' not found in trace database. "
            f"Available: {list(trace_set.definitions.keys())}"
        )
    definition = trace_set.definitions[solution.definition]

    if shutil.which(rocprof_path) is None:
        return f"ERROR: '{rocprof_path}' not found. Install ROCm rocprofiler-sdk (rocprofv3)."

    with tempfile.TemporaryDirectory(prefix="fib_rocprof_", dir=tmpdir) as tmp:
        tmp_path = Path(tmp)
        data_dir = tmp_path / "data"
        out_dir = tmp_path / "out"
        data_dir.mkdir()
        out_dir.mkdir()  # rocprofv3 -d target; create explicitly rather than rely on the tool
        (data_dir / "definition.json").write_text(definition.model_dump_json())
        (data_dir / "solution.json").write_text(solution.model_dump_json())
        (data_dir / "workload.json").write_text(workload.model_dump_json())

        cmd = _build_cmd(
            data_dir,
            out_dir,
            device,
            Path(trace_set_path) if trace_set_path else None,
            counters,
            rocprof_path,
        )
        env = os.environ.copy()
        if tmpdir:
            env["TMPDIR"] = tmpdir

        logger.info("FlashInfer Bench rocprof: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"ERROR: rocprofv3 timed out after {timeout} seconds."
        except OSError as e:  # on PATH but not launchable (permissions, bad interpreter)
            return f"ERROR: failed to launch rocprofv3: {e}"

        if result.returncode != 0:
            # A failing rocprofv3 can emit a very large stdout/stderr, so honour max_lines here
            # too. Only the payload is truncated — the "ERROR:" prefix must survive any limit.
            detail = truncate(f"{result.stdout}\n{result.stderr}", max_lines)
            return f"ERROR: rocprofv3 exited with code {result.returncode}:\n{detail}"

        return _format_kernel_report(out_dir, max_lines)
