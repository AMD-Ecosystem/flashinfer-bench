"""P1 end-to-end proof: an AITER-backed solution through the real Benchmark loop on ROCm.

Builds an in-memory TraceSet (RMSNorm definition + two solutions: an AITER-backed one and a
pure-torch one) and runs the real `Benchmark` path, which builds each solution, times it with the
ROCm timing backend, checks correctness vs the reference, and computes speedup.

Run inside the container:  python docker/rocm/validate_aiter_rmsnorm.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Definition,
    EvaluationStatus,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
    save_json_file,
    save_jsonl_file,
)

H = 4096
EPS = 1e-6

# RMSNorm reference: out = (x / rms(x)) * weight, computed in fp32 then cast (fp16-friendly).
REFERENCE = f"""
import torch

def run(x, weight):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    normed = (xf * torch.rsqrt(var + {EPS})).to(x.dtype)
    return normed * weight
"""

# AITER-backed solution: a thin Python wrapper calling the tuned AITER op. This is exactly the
# shape the §3.9 generator will emit automatically for covered op-types.
AITER_SOLUTION = f"""
import aiter

def run(x, weight):
    return aiter.rms_norm(x, weight, {EPS})
"""

# Pure-torch solution (sanity baseline; equivalent to the reference).
TORCH_SOLUTION = f"""
import torch

def run(x, weight):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    normed = (xf * torch.rsqrt(var + {EPS})).to(x.dtype)
    return normed * weight
"""


def _solution(name: str, entry_file: str, content: str) -> Solution:
    return Solution(
        name=name,
        definition="rmsnorm_h4096",
        author="rocm-p1",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["rocm"],
            entry_point=f"{entry_file}::run",
            destination_passing_style=False,
        ),
        sources=[SourceFile(path=entry_file, content=content)],
    )


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
            reference=REFERENCE,
        )
        save_json_file(definition, root / "definitions" / "rmsnorm_h4096.json")

        for sol in (
            _solution("rmsnorm_aiter", "aiter_impl.py", AITER_SOLUTION),
            _solution("rmsnorm_torch", "torch_impl.py", TORCH_SOLUTION),
        ):
            save_json_file(sol, root / "solutions" / f"{sol.name}.json")

        workload = Workload(
            axes={"M": 4096},
            inputs={"x": RandomInput(), "weight": RandomInput()},
            uuid="rmsnorm_wl",
        )
        save_jsonl_file(
            [Trace(definition="rmsnorm_h4096", workload=workload)],
            root / "workloads" / "op" / "rmsnorm_h4096.jsonl",
        )

        trace_set = TraceSet.from_path(str(root))
        config = BenchmarkConfig(warmup_runs=5, iterations=20, num_trials=1)
        result = benchmark = Benchmark(trace_set, config).run_all(dump_traces=False)

        traces = result.traces.get("rmsnorm_h4096", [])
        print("=" * 72)
        print("AITER RMSNorm end-to-end (real Benchmark loop) on ROCm")
        print("=" * 72)
        ok = False
        for t in traces:
            ev = t.evaluation
            status = ev.status if ev else "NO-EVAL"
            perf = getattr(ev, "performance", None) if ev else None
            lat = f"{perf.latency_ms:.5f} ms" if perf else "n/a"
            spd = f"{perf.speedup_factor:.2f}x" if perf else "n/a"
            print(f"[{status}] {t.solution:<16} latency={lat:<14} speedup={spd}")
            if t.solution == "rmsnorm_aiter" and status == EvaluationStatus.PASSED:
                ok = True
        print("-" * 72)
        print("RESULT:", "PASS — AITER solution built, correct, benchmarked" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
