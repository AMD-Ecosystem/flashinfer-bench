"""P1 end-to-end proof: AITER-generated solutions for every supported op-type on ROCm.

For each op-type the generator supports, builds an in-memory definition + workload, asks the §3.9
generator for the AITER-backed solution, and runs it through the real Benchmark loop (build → time →
correctness → speedup) on the GPU.

Run inside the container:  python docker/rocm/validate_aiter_ops.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    Definition,
    EvaluationStatus,
    RandomInput,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
    save_json_file,
    save_jsonl_file,
)
from flashinfer_bench.integration.aiter import generate_aiter_solution, is_aiter_available

H = 4096

RMSNORM_REF = """
import torch

def run(x, weight):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + 1e-6)).to(x.dtype) * weight
"""

LAYERNORM_REF = """
import torch

def run(x, weight, bias):
    out = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), 1e-5)
    return out.to(x.dtype)
"""

SILU_MUL_REF = """
import torch

def run(x):
    a, b = x.chunk(2, dim=-1)
    return (torch.nn.functional.silu(a.float()) * b.float()).to(x.dtype)
"""


def _defs():
    return [
        Definition(
            name="rmsnorm_h4096",
            op_type="rmsnorm",
            axes={"M": AxisVar(), "H": AxisConst(value=H)},
            inputs={
                "x": TensorSpec(shape=["M", "H"], dtype="float16"),
                "weight": TensorSpec(shape=["H"], dtype="float16"),
            },
            outputs={"out": TensorSpec(shape=["M", "H"], dtype="float16")},
            reference=RMSNORM_REF,
        ),
        Definition(
            name="layernorm_h4096",
            op_type="layernorm",
            axes={"M": AxisVar(), "H": AxisConst(value=H)},
            inputs={
                "x": TensorSpec(shape=["M", "H"], dtype="float16"),
                "weight": TensorSpec(shape=["H"], dtype="float16"),
                "bias": TensorSpec(shape=["H"], dtype="float16"),
            },
            outputs={"out": TensorSpec(shape=["M", "H"], dtype="float16")},
            reference=LAYERNORM_REF,
        ),
        Definition(
            name="silu_and_mul_h4096",
            op_type="silu_and_mul",
            axes={"M": AxisVar(), "H": AxisConst(value=H), "H2": AxisConst(value=2 * H)},
            inputs={"x": TensorSpec(shape=["M", "H2"], dtype="float16")},
            outputs={"out": TensorSpec(shape=["M", "H"], dtype="float16")},
            reference=SILU_MUL_REF,
        ),
    ]


def main() -> int:
    if not is_aiter_available():
        print("aiter not available; cannot validate")
        return 1

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        definitions = _defs()
        generated = []
        for defn in definitions:
            save_json_file(defn, root / "definitions" / f"{defn.name}.json")
            sol = generate_aiter_solution(defn)
            if sol is None:
                print(f"[skip] no AITER solution generated for {defn.op_type}")
                continue
            generated.append(sol.name)
            save_json_file(sol, root / "solutions" / f"{sol.name}.json")
            wl = Workload(
                axes={"M": 4096}, inputs={k: RandomInput() for k in defn.inputs}, uuid=f"{defn.name}_wl"
            )
            save_jsonl_file(
                [Trace(definition=defn.name, workload=wl)],
                root / "workloads" / "op" / f"{defn.name}.jsonl",
            )

        trace_set = TraceSet.from_path(str(root))
        config = BenchmarkConfig(warmup_runs=5, iterations=20, num_trials=1)
        result = Benchmark(trace_set, config).run_all(dump_traces=False)

        print("=" * 76)
        print("AITER-generated solutions — end-to-end (real Benchmark loop) on ROCm")
        print("=" * 76)
        n_pass = 0
        for defn in definitions:
            for t in result.traces.get(defn.name, []):
                ev = t.evaluation
                status = ev.status if ev else "NO-EVAL"
                perf = getattr(ev, "performance", None) if ev else None
                lat = f"{perf.latency_ms:.5f} ms" if perf else "n/a"
                spd = f"{perf.speedup_factor:.2f}x" if perf else "n/a"
                passed = status == EvaluationStatus.PASSED
                n_pass += passed
                print(f"[{status}] {defn.op_type:<14} {t.solution:<26} latency={lat:<14} speedup={spd}")
        print("-" * 76)
        print(f"RESULT: {n_pass}/{len(generated)} generated AITER solutions PASSED")
        return 0 if n_pass == len(generated) and generated else 1


if __name__ == "__main__":
    sys.exit(main())
