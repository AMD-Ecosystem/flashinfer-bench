"""P0 in-container validation for the ROCm port.

Run inside the container:  python docker/rocm/validate_p0.py
Exercises the full ROCm stack end-to-end and reports PASS/FAIL per check.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            detail = fn() or ""
            results.append((name, True, str(detail)))
        except Exception as e:  # noqa: BLE001
            results.append((name, False, f"{e.__class__.__name__}: {e}"))
            if os.environ.get("FIB_VALIDATE_VERBOSE"):
                traceback.print_exc()
        return fn

    return deco


@check("torch + ROCm (torch.cuda on gfx942)")
def _():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    name = torch.cuda.get_device_name(0)
    arch = torch.cuda.get_device_properties(0).gcnArchName
    return f"torch={torch.__version__} hip={torch.version.hip} dev='{name}' arch={arch}"


@check("torch GPU compute (matmul)")
def _():
    import torch

    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = (a @ b).float().sum().item()
    torch.cuda.synchronize()
    return f"matmul ok, checksum finite={c == c}"


@check("import flashinfer (amd-flashinfer)")
def _():
    import flashinfer

    return f"flashinfer={getattr(flashinfer, '__version__', 'unknown')}"


@check("import aiter")
def _():
    import aiter

    return f"aiter={getattr(aiter, '__version__', 'unknown')}"


@check("tvm-ffi HIP backend detection")
def _():
    import tvm_ffi.cpp.extension as ext

    # Private but stable across 0.1.12; used only for validation.
    detect = getattr(ext, "_detect_gpu_backend", None) or getattr(ext, "_detect_backend", None)
    backend = detect() if detect else "n/a"
    return f"tvm_ffi detected backend={backend}"


@check("tvm-ffi HIP end-to-end compile + run")
def _():
    import torch
    import tvm_ffi

    src = r"""
#include <hip/hip_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

namespace ffi = tvm::ffi;

__global__ void add_one_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

void add_one(ffi::TensorView x) {
    int n = 1;
    for (int i = 0; i < x.ndim(); ++i) n *= x.shape()[i];
    int block = 256, grid = (n + block - 1) / block;
    add_one_kernel<<<grid, block>>>(static_cast<float*>(x.data_ptr()), n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, add_one);
"""
    with tempfile.TemporaryDirectory() as d:
        cu = os.path.join(d, "add_one.cu")
        with open(cu, "w") as f:
            f.write(src)
        lib = tvm_ffi.cpp.build(
            name="fib_p0_addone", cpp_files=[], cuda_files=[cu], build_directory=d
        )
        mod = tvm_ffi.load_module(lib)
        t = torch.ones(1024, device="cuda", dtype=torch.float32)
        mod.add_one(t)
        torch.cuda.synchronize()
        ok = bool((t == 2.0).all().item())
        if not ok:
            raise RuntimeError("add_one kernel did not increment tensor")
        return f"compiled+ran HIP kernel via tvm-ffi at {os.path.basename(lib)}"


@check("torch nvtx -> roctx mapping")
def _():
    import torch

    # If this maps to roctx, rocprofv3 --marker-trace will capture it; here we
    # just confirm the API exists and is callable on ROCm.
    torch.cuda.nvtx.range_push("flashinfer_bench_ncu_profile")
    torch.cuda.nvtx.range_pop()
    return "torch.cuda.nvtx range push/pop callable"


@check("import flashinfer_bench")
def _():
    import flashinfer_bench as flb

    return f"flashinfer_bench={getattr(flb, '__version__', 'unknown')}"


def main() -> int:
    print("=" * 72)
    print("FlashInfer-Bench ROCm P0 validation")
    print("=" * 72)
    width = max(len(n) for n, _, _ in results) if results else 40
    npass = 0
    for name, ok, detail in results:
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name.ljust(width)}  {detail}")
        npass += ok
    print("-" * 72)
    print(f"{npass}/{len(results)} checks passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
