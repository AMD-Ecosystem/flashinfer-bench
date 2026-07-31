"""P3 proof: the rocprofv3-based agent profiler works end-to-end on ROCm.

Profiles a solution with flashinfer_bench_run_rocprof and checks it returns per-kernel device
timing + occupancy resources scoped to the runner's roctx region.

Run inside the container:  python3 docker/rocm/validate_rocprof.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from flashinfer_bench.agents import (
    flashinfer_bench_list_rocprof_options,
    flashinfer_bench_run_rocprof,
)
from flashinfer_bench.agents._profiling import PROFILE_REGION
from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    save_json_file,
)
from flashinfer_bench.data.workload import RandomInput, Workload

H = 4096

# Destination-passing RMSNorm (the runner calls call_destination_passing(*inputs, *outputs)).
DPS_SRC = """
import torch

def run(x, weight, out):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    out.copy_((xf * torch.rsqrt(var + 1e-6)).to(x.dtype) * weight)
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        definition = Definition(
            name="rmsnorm_h4096",
            op_type="rmsnorm",
            axes={"M": AxisVar(), "H": AxisConst(value=H)},
            inputs={
                "x": TensorSpec(shape=["M", "H"], dtype="float16"),
                "weight": TensorSpec(shape=["H"], dtype="float16"),
            },
            outputs={"out": TensorSpec(shape=["M", "H"], dtype="float16")},
            reference="import torch\n\ndef run(x, weight):\n    return x * weight\n",
        )
        save_json_file(definition, root / "definitions" / "rmsnorm_h4096.json")

        solution = Solution(
            name="rmsnorm_dps",
            definition="rmsnorm_h4096",
            author="rocm-p3",
            spec=BuildSpec(
                language=SupportedLanguages.PYTHON,
                target_hardware=["rocm"],
                entry_point="impl.py::run",
                destination_passing_style=True,
            ),
            sources=[SourceFile(path="impl.py", content=DPS_SRC)],
        )
        workload = Workload(
            axes={"M": 4096}, inputs={"x": RandomInput(), "weight": RandomInput()}, uuid="wl"
        )

        print("=" * 72)
        print("rocprofv3 agent profiler — end-to-end on ROCm")
        print("=" * 72)

        opts = flashinfer_bench_list_rocprof_options()
        print("list_rocprof_options:", "OK" if not opts.startswith("ERROR") else opts[:120])

        report = flashinfer_bench_run_rocprof(
            solution, workload, device="cuda:0", trace_set_path=str(root), max_lines=25
        )
        print("-" * 72)
        print(report)
        print("-" * 72)
        # "us |" alone is far too weak: it passed while region correlation was broken and the
        # report contained every kernel in the process (torch init, RNG, copies). Require the
        # header to state real scoping, which is the whole point of profiling under a roctx range.
        problems = []
        if report.startswith("ERROR"):
            problems.append("tool returned an error")
        if "us |" not in report:
            problems.append("no kernel timing lines")
        if "ALL kernels" in report:
            problems.append("region correlation failed — report is unscoped")
        elif f"region '{PROFILE_REGION}'" not in report:
            problems.append("header does not report the profiled region")

        ok = not problems
        print(
            "RESULT:",
            (
                "PASS — rocprofv3 profiled kernels, scoped to the region"
                if ok
                else "FAIL — " + "; ".join(problems)
            ),
        )
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
