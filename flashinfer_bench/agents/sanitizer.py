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
from typing import List, Literal, Optional, Tuple, Union

from flashinfer_bench.data import Solution, TraceSet, Workload

from ._output import truncate

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


def _xnack_status(run_env) -> str:
    """Describe the fault-detection actually in effect, so a clean verdict can be read correctly.

    A "no fault detected" result means much less on an ``xnack-`` target, where page-fault-based
    detection is unavailable no matter what ``HSA_XNACK`` is set to.
    """
    value = run_env.get("HSA_XNACK", "<unset>")
    arch = ""
    try:
        import torch

        if torch.cuda.is_available():
            arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # torch missing, no device, or a driver hiccup — arch is a nice-to-have
        pass

    if "xnack-" in arch:
        return (
            f"HSA_XNACK={value}, but the device target is {arch}: page-fault-based detection is "
            "NOT active, so only faults that abort the process were catchable."
        )
    if arch:
        return f"HSA_XNACK={value}, device target {arch}."
    return f"HSA_XNACK={value} (device target undetermined)."


def _run_memcheck(
    data_dir: Path, device: str, trace_set_path: Optional[Path], timeout: int, env
) -> Tuple[str, str]:
    """Best-effort memcheck: run the solution and detect GPU memory faults.

    Returns
    -------
    log : str
        The runner's stdout/stderr and return code. Unbounded (hundreds of lines for a JIT build);
        the caller is what truncates it.
    verdict : str
        The conclusion — "MEMCHECK: ..." lines, or an "ERROR:"-prefixed tool failure. Returned
        *separately* from ``log`` so truncation can never drop it: ``truncate`` keeps the FIRST
        lines, so a verdict appended to a chatty log would be cut off entirely and a detected
        memory fault would come back looking like an ordinary build log.
    """
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
    # HSA_XNACK=1 asks for page-fault-based detection of out-of-bounds device accesses (surfacing
    # faults instead of silently reading garbage). setdefault, NOT a forced assignment: code
    # objects on CDNA are built per xnack variant (e.g. gfx942:xnack-), so overriding an explicit
    # caller setting can mismatch the target. The verdict reports what was actually in effect
    # instead — see _xnack_status.
    run_env = dict(env)
    run_env.setdefault("HSA_XNACK", "1")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=run_env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "", f"ERROR: memcheck run timed out after {timeout} seconds."
    except Exception as e:  # e.g. the runner failed to launch (exec/permission)
        return "", f"ERROR: memcheck failed to launch the solution runner: {e}"

    log = f"STDOUT:\n{r.stdout}\n\nSTDERR:\n{r.stderr}\nReturn code: {r.returncode}\n"
    faults = [s for s in _MEM_FAULT_SIGNATURES if s.lower() in (r.stdout + r.stderr).lower()]
    if faults:
        return log, f"\nMEMCHECK: FAIL — detected fault signatures: {faults}\n"
    if r.returncode != 0:
        # A non-zero exit with no fault signature is a build error, a bad device string, or a
        # Python exception in the runner — a tool failure, not a memcheck verdict. Reporting it
        # as "MEMCHECK: FAIL" would send an agent hunting for a memory bug that isn't there.
        return log, (
            f"ERROR: the solution runner exited with code {r.returncode} without any GPU "
            f"memory-fault signature, so this is a run failure rather than a memcheck result."
        )
    return log, (
        "\nMEMCHECK: no GPU memory fault detected.\n"
        f"DETECTION: {_xnack_status(run_env)}\n"
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
        Truncate each check's runner log to this many lines. The verdict ("MEMCHECK: ..." or
        "ERROR: ...") and the section headers are always kept, so no limit can hide a detected
        fault.

    Returns
    -------
    str
        Results text, or an error string starting with "ERROR:".
    """
    if sanitizer_types is None:
        sanitizer_types = ["memcheck"]
    for st in sanitizer_types:
        if st not in VALID_SANITIZER_TYPES:
            # sorted(): a bare set repr reorders per process (string hash randomization), which
            # makes this message unstable for agents that parse or diff tool output.
            valid = ", ".join(sorted(VALID_SANITIZER_TYPES))
            return f"ERROR: Invalid sanitizer type '{st}'. Must be one of: {valid}"

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
            log, verdict = _run_memcheck(
                data_dir, device, Path(trace_set_path) if trace_set_path else None, timeout, env
            )
            # Preserve the agent-tool contract: an error must be returned as a string that
            # *starts* with "ERROR:", so short-circuit instead of burying it in section output.
            if verdict.startswith("ERROR:"):
                # Honour max_lines on this path too — the runner log can be very large. Only the
                # log is truncated, so the "ERROR:" line survives any limit (max_lines=0 included).
                return f"{verdict}\n{truncate(log, max_lines)}" if log else verdict
            # Same reasoning for the success path: truncate only the log and keep the verdict
            # whole. `truncate` keeps the FIRST lines, so folding the verdict into the truncated
            # text would let a chatty run drop a "MEMCHECK: FAIL" and hand back a real memory
            # fault as what reads like an ordinary build log.
            out += truncate(log, max_lines) + verdict

        out += f"\n{'=' * 60}\nSanitizer checks complete\n{'=' * 60}\n"
        return out
