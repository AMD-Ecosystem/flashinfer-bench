"""P2 proof: a C++/CUDA (tvm-ffi) solution builds and runs on ROCm via the real loop.

Exercises TVMFFIBuilder end-to-end: a .cu kernel with the default tvm-ffi binding is compiled (tvm-ffi
auto-detects the HIP backend -> hipcc, --offload-arch, -lamdhip64) and benchmarked against the
Python reference on the GPU. Validates the P2 change that removed the hardcoded CUDA link flags.

Run inside the container:  python docker/rocm/validate_cpp_tvmffi.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from flashinfer_bench.bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import (
    AxisVar,
    BuildSpec,
    Definition,
    EvaluationStatus,
    RandomInput,
    Solution,
    SourceFile,
    SupportedBindings,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
    save_json_file,
    save_jsonl_file,
)

# Destination-passing tvm-ffi kernel: y = x + 1. Uses HIP includes; on ROCm tvm-ffi compiles the
# .cu with hipcc (-x hip). TensorView API: .ndim(), .shape(), .data_ptr().
CUDA_SRC = r"""
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

namespace ffi = tvm::ffi;

__global__ void add_one_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + 1.0f;
}

void run(ffi::TensorView x, ffi::TensorView y) {
    int n = 1;
    for (int i = 0; i < x.ndim(); ++i) n *= x.shape()[i];
    int block = 256, grid = (n + block - 1) / block;
    add_one_kernel<<<grid, block>>>(
        static_cast<const float*>(x.data_ptr()),
        static_cast<float*>(y.data_ptr()), n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
"""

REFERENCE = "import torch\n\ndef run(x):\n    return x + 1.0\n"


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        definition = Definition(
            name="add_one_n",
            op_type="elementwise",
            axes={"N": AxisVar()},
            inputs={"x": TensorSpec(shape=["N"], dtype="float32")},
            outputs={"y": TensorSpec(shape=["N"], dtype="float32")},
            reference=REFERENCE,
        )
        save_json_file(definition, root / "definitions" / "add_one_n.json")

        solution = Solution(
            name="add_one_tvmffi",
            definition="add_one_n",
            author="rocm-p2",
            spec=BuildSpec(
                language=SupportedLanguages.CUDA,
                target_hardware=["rocm"],
                entry_point="add_one.cu::run",
                binding=SupportedBindings.TVM_FFI,
                destination_passing_style=True,
            ),
            sources=[SourceFile(path="add_one.cu", content=CUDA_SRC)],
        )
        save_json_file(solution, root / "solutions" / "add_one_tvmffi.json")

        wl = Workload(axes={"N": 1 << 20}, inputs={"x": RandomInput()}, uuid="add_one_wl")
        save_jsonl_file(
            [Trace(definition="add_one_n", workload=wl)],
            root / "workloads" / "op" / "add_one_n.jsonl",
        )

        trace_set = TraceSet.from_path(str(root))
        config = BenchmarkConfig(warmup_runs=3, iterations=10, num_trials=1)
        result = Benchmark(trace_set, config).run_all(dump_traces=False)

        print("=" * 72)
        print("C++/CUDA (tvm-ffi) solution end-to-end on ROCm")
        print("=" * 72)
        ok = False
        for t in result.traces.get("add_one_n", []):
            ev = t.evaluation
            status = ev.status if ev else "NO-EVAL"
            perf = getattr(ev, "performance", None) if ev else None
            lat = f"{perf.latency_ms:.5f} ms" if perf else "n/a"
            print(f"[{status}] {t.solution:<18} latency={lat}")
            if status == EvaluationStatus.PASSED:
                ok = True
            elif ev and getattr(ev, "error_message", None):
                print("  error:", str(ev.error_message)[:300])
        print("-" * 72)
        print("RESULT:", "PASS — tvm-ffi HIP C++ solution built & correct" if ok else "FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
