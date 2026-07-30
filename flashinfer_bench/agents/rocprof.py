"""rocprofv3-based profiling tool for LLM agents (ROCm replacement for NCU).

Runs a solution under ``rocprofv3`` and reports per-kernel device-side timing and occupancy-relevant
resource usage (VGPR/SGPR/LDS/scratch, grid/block), scoped to the profiled region marked by the
runner's roctx range. Optionally collects hardware performance counters via ``--pmc``.

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

logger = logging.getLogger(__name__)

# roctx region emitted by _solution_runner around the profiled (non-warmup) run.
_PROFILE_REGION = "flashinfer_bench_ncu_profile"


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


def _region_window(out_dir: Path) -> Optional[tuple]:
    """Return (start_ns, end_ns) of the profiled roctx region, or None if not found."""
    rows = _read_csv(str(out_dir / "**" / "*marker_api_trace*.csv"))
    for r in rows:
        if _PROFILE_REGION in (r.get("Name") or ""):
            try:
                return int(r["Start_Timestamp"]), int(r["End_Timestamp"])
            except (KeyError, TypeError, ValueError):
                # Multi-process runs emit several marker CSVs (all merged by _read_csv), so a
                # malformed row is not the last word — keep scanning for a usable one rather
                # than silently giving up on region scoping.
                continue
    return None


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

    window = _region_window(out_dir)
    selected = [
        (r, end - start)
        for r, start, end in parsed
        if window is None or (start >= window[0] and end <= window[1])
    ]
    # Fall back to all kernels if region correlation found nothing.
    if not selected:
        selected = [(r, end - start) for r, start, end in parsed]

    selected.sort(key=lambda x: x[1], reverse=True)
    lines = [
        f"rocprofv3 kernel profile (region '{_PROFILE_REGION}', "
        f"{len(selected)} kernel dispatch(es), sorted by duration):",
        "",
    ]
    for r, dur_ns in selected:
        name = (r.get("Kernel_Name") or "?").strip('"')
        if len(name) > 80:
            name = name[:77] + "..."
        lines.append(
            f"  {dur_ns / 1000.0:9.3f} us | VGPR {r.get('VGPR_Count','?'):>4} "
            f"SGPR {r.get('SGPR_Count','?'):>4} LDS {r.get('LDS_Block_Size','?'):>6} "
            f"scratch {r.get('Scratch_Size','?'):>6} | "
            f"grid {r.get('Grid_Size_X','?')}x{r.get('Grid_Size_Y','?')}x{r.get('Grid_Size_Z','?')} "
            f"block {r.get('Workgroup_Size_X','?')}x{r.get('Workgroup_Size_Y','?')}x{r.get('Workgroup_Size_Z','?')}"
        )
        lines.append(f"    {name}")

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
