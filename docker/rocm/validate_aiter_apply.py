"""P1 proof: AITER solutions flow through the `apply` runtime path on ROCm (§3.9 step 3).

Flow: build a dataset with an RMSNorm definition -> augment it with an AITER solution via
`augment_trace_set_with_aiter` -> benchmark to produce traces -> build ApplyRuntime and confirm
`dispatch` routes to the AITER solution (fallback raises if it were used) and returns correct output.

Run inside the container:  python docker/rocm/validate_aiter_apply.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from flashinfer_bench.apply import ApplyConfig
from flashinfer_bench.apply.runtime import ApplyRuntime
from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    Definition,
    RandomInput,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
    save_json_file,
    save_jsonl_file,
)
from flashinfer_bench.integration.aiter import augment_trace_set_with_aiter, is_aiter_available

H = 4096
M = 4096

REF = """
import torch

def run(x, weight):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + 1e-6)).to(x.dtype) * weight
"""


def _ref(x, weight):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + 1e-6)).to(x.dtype) * weight


def main() -> int:
    if not is_aiter_available():
        print("aiter not available; cannot validate")
        return 1

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        defn = Definition(
            name="rmsnorm_h4096",
            op_type="rmsnorm",
            axes={"M": AxisVar(), "H": AxisConst(value=H)},
            inputs={
                "x": TensorSpec(shape=["M", "H"], dtype="float16"),
                "weight": TensorSpec(shape=["H"], dtype="float16"),
            },
            outputs={"out": TensorSpec(shape=["M", "H"], dtype="float16")},
            reference=REF,
        )
        save_json_file(defn, root / "definitions" / "rmsnorm_h4096.json")
        wl = Workload(
            axes={"M": M}, inputs={"x": RandomInput(), "weight": RandomInput()}, uuid="wl"
        )
        save_jsonl_file(
            [Trace(definition="rmsnorm_h4096", workload=wl)],
            root / "workloads" / "op" / "rmsnorm_h4096.jsonl",
        )

        trace_set = TraceSet.from_path(str(root))

        # 1) AITER becomes a candidate in the dataset.
        n = augment_trace_set_with_aiter(trace_set)
        print(f"augment_trace_set_with_aiter -> added {n} solution(s)")

        # 2) Benchmark to produce traces, then feed them back so the apply table can rank them.
        result = Benchmark(
            trace_set, BenchmarkConfig(warmup_runs=5, iterations=20, num_trials=1)
        ).run_all(dump_traces=False)
        traces = [t for lst in result.traces.values() for t in lst]
        trace_set.add_traces(traces)

        # 3) apply runtime routes to the best (AITER) solution.
        #    - use_def_best: prefer the top-ranked tuned solution when a workload key isn't an exact
        #      indexed match.
        #    - tolerances: ApplyConfig.filter_traces gates on max_absolute_error AND
        #      max_relative_error (defaults 1e-2 / 1e-5). For fp16 elementwise ops like RMSNorm the
        #      *relative* error is large near-zero outputs (here ~0.16 with abs err ~0.008), so we
        #      set an op-appropriate max_rtol. NOTE: apply's max-relative-error filter differs from
        #      the benchmark's combined/matched-ratio correctness and is overly strict for such ops
        #      (flagged in ROCM_PORT_PLAN.md as a follow-up).
        rt = ApplyRuntime(
            trace_set, ApplyConfig(on_miss_policy="use_def_best", max_atol=1e-2, max_rtol=1.0)
        )
        best = rt._table.def_best.get("rmsnorm_h4096")
        print(f"apply table def_best[rmsnorm_h4096] = {best}")

        x = torch.randn(M, H, device="cuda", dtype=torch.float16)
        weight = torch.randn(H, device="cuda", dtype=torch.float16)

        def _fallback(*_a, **_k):
            raise RuntimeError("fallback used — apply did NOT route to a solution")

        out = rt.dispatch("rmsnorm_h4096", args=(x, weight), kwargs={}, fallback=_fallback)
        err = (out.float() - _ref(x, weight).float()).abs().max().item()

        used_aiter = best == "rmsnorm_h4096__aiter"
        correct = err < 0.05
        print(f"dispatch: max_abs_err={err:.4f}, routed_to_aiter={used_aiter}")
        print("-" * 60)
        ok = used_aiter and correct
        print("RESULT:", "PASS — apply routed to the AITER solution" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
