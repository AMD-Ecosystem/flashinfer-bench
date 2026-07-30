"""Memory/correctness checking tool for LLM agents (ROCm best-effort).

NVIDIA's compute-sanitizer has no full ROCm equivalent. This tool provides a best-effort
"memcheck" by running the solution and detecting GPU memory faults reported by the HIP runtime
(illegal address / page fault / HSA memory fault). The race/sync/init sub-tools have no ROCm
counterpart and return a clear "unsupported" message rather than failing, so agents degrade
gracefully. The JSON-serializable, "ERROR:"-prefixed contract matches the previous tool.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Literal, Optional, Union

from flashinfer_bench.data import Solution, TraceSet, Workload

logger = logging.getLogger(__name__)

SanitizerType = Literal["memcheck", "racecheck", "initcheck", "synccheck"]
VALID_SANITIZER_TYPES: set[SanitizerType] = {"memcheck", "racecheck", "initcheck", "synccheck"}

# Sub-tools with no ROCm equivalent (compute-sanitizer racecheck/initcheck/synccheck).
_UNSUPPORTED_ON_ROCM: set[SanitizerType] = {"racecheck", "initcheck", "synccheck"}

# HIP/HSA runtime signatures indicating a GPU memory fault.
_MEM_FAULT_SIGNATURES = (
    "Memory access fault",
    "HSA_STATUS_ERROR_MEMORY_FAULT",
    "page fault",
    "hipErrorIllegalAddress",
    "an illegal memory access",
)


def _truncate_output(output: str, max_lines: int) -> str:
    lines = output.split("\n")
    if len(lines) <= max_lines:
        return output
    return "\n".join(lines[:max_lines]) + f"\n[... {len(lines) - max_lines} more lines]"


def _run_memcheck(
    data_dir: Path, device: str, trace_set_path: Optional[Path], timeout: int, env
) -> str:
    """Best-effort memcheck: run the solution and detect GPU memory faults."""
    cmd = [
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
        cmd += ["--trace-set-path", str(trace_set_path)]
    # HSA_XNACK=1 enables page-fault-based detection of out-of-bounds device accesses where the
    # hardware/driver supports it (surfaces faults instead of silently reading garbage).
    run_env = dict(env)
    run_env.setdefault("HSA_XNACK", "1")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=run_env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"ERROR: memcheck run timed out after {timeout} seconds."
    except Exception as e:  # e.g. the runner failed to launch (exec/permission)
        return f"ERROR: memcheck failed to launch the solution runner: {e}"

    combined = f"STDOUT:\n{r.stdout}\n\nSTDERR:\n{r.stderr}\nReturn code: {r.returncode}\n"
    faults = [s for s in _MEM_FAULT_SIGNATURES if s.lower() in (r.stdout + r.stderr).lower()]
    if faults:
        return combined + f"\nMEMCHECK: FAIL — detected fault signatures: {faults}\n"
    if r.returncode != 0:
        # A non-zero exit with no fault signature is a build error, a bad device string, or a
        # Python exception in the runner — a tool failure, not a memcheck verdict. Reporting it
        # as "MEMCHECK: FAIL" would send an agent hunting for a memory bug that isn't there.
        return (
            f"ERROR: the solution runner exited with code {r.returncode} without any GPU "
            f"memory-fault signature, so this is a run failure rather than a memcheck result.\n"
            + combined
        )
    return combined + (
        "\nMEMCHECK: no GPU memory fault detected.\n"
        "NOTE: ROCm has no full compute-sanitizer equivalent; this only catches faults that abort "
        "the process (illegal address / page fault). It does NOT detect benign OOB reads, "
        "uninitialized memory, or races. For deeper checks, build the kernel with ROCm's LLVM "
        "AddressSanitizer (-fsanitize=address, xnack) or inspect with rocgdb.\n"
    )


def flashinfer_bench_run_sanitizer(
    solution: Union[Solution, str],
    workload: Union[Workload, str],
    *,
    device: str = "cuda:0",
    trace_set_path: Optional[str] = None,
    sanitizer_types: Optional[List[SanitizerType]] = None,
    timeout: int = 300,
    tmpdir: Optional[str] = None,
    max_lines: Optional[int] = None,
) -> str:
    """Run best-effort memory checks on a solution+workload (ROCm).

    Parameters
    ----------
    solution : Solution or str
        Solution object or path to a solution JSON file.
    workload : Workload or str
        Workload object or path to a workload JSON file.
    device : str
        Device to run on ("cuda" device string on ROCm). Default "cuda:0".
    trace_set_path : str, optional
        Path to the trace set. Defaults to FIB_DATASET_PATH.
    sanitizer_types : List[SanitizerType], optional
        Which checks to run. Default ["memcheck"]. On ROCm only "memcheck" is supported
        (best-effort); "racecheck"/"initcheck"/"synccheck" report as unsupported.
    timeout : int
        Timeout in seconds per check. Default 300.
    tmpdir : str, optional
        Temporary directory.
    max_lines : int, optional
        Truncate output to this many lines.

    Returns
    -------
    str
        Results text, or an error string starting with "ERROR:".
    """
    if sanitizer_types is None:
        sanitizer_types = ["memcheck"]
    for st in sanitizer_types:
        if st not in VALID_SANITIZER_TYPES:
            return f"ERROR: Invalid sanitizer type '{st}'. Must be one of: {VALID_SANITIZER_TYPES}"

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

    with tempfile.TemporaryDirectory(prefix="fib_sanitizer_", dir=tmpdir) as tmp:
        data_dir = Path(tmp)
        (data_dir / "definition.json").write_text(definition.model_dump_json())
        (data_dir / "solution.json").write_text(solution.model_dump_json())
        (data_dir / "workload.json").write_text(workload.model_dump_json())

        env = os.environ.copy()
        if tmpdir:
            env["TMPDIR"] = tmpdir

        out = ""
        for st in sanitizer_types:
            out += f"\n{'=' * 60}\n{st.upper()}\n{'=' * 60}\n\n"
            if st in _UNSUPPORTED_ON_ROCM:
                out += (
                    f"{st} is not supported on ROCm: NVIDIA compute-sanitizer's {st} has no AMD "
                    "equivalent. Skipped. (memcheck is available as a best-effort fault detector.)\n"
                )
                continue
            result = _run_memcheck(
                data_dir, device, Path(trace_set_path) if trace_set_path else None, timeout, env
            )
            # Preserve the agent-tool contract: an error must be returned as a string that
            # *starts* with "ERROR:", so short-circuit instead of burying it in section output.
            if result.startswith("ERROR:"):
                return result
            out += result

        out += f"\n{'=' * 60}\nSanitizer checks complete\n{'=' * 60}\n"
        if max_lines is not None:
            out = _truncate_output(out, max_lines)
        return out
