---
name: author-hip-solution
description: Write hand-authored C++/HIP bench Solutions that build on ROCm via the torch and tvm-ffi builders. Full CUDA→ROCm porting cheat sheet (at::Tensor, HIP stream/guard masquerade, TORCH_LIBRARY vs PYBIND11, dispatch/validation macros), gpu_iface divergence pattern, -ffast-math/finite-math gotcha, arch targeting, wavefront=64, FP8 _fnuz, and CDNA3-vs-CDNA4 differences. Use when AITER and Triton don't cover an op and you need a native kernel solution on gfx942/gfx950.
---

# Author a HIP Solution

For ops that neither [AITER](../generate-aiter-solution/SKILL.md) nor Triton cover, write a
`language=cuda` bench `Solution` that compiles to a HIP `.so` on ROCm. On AMD the C++ builders route
through **hipcc** and (for tvm-ffi) hipify `.cu` at build time — so a well-written `.cu` builds
mostly unchanged; the work is avoiding CUDA-only assumptions. This skill is the CUDA→ROCm porting
know-how; the bench builders are the mechanism.

Builders: `compile/builders/torch_builder.py` (auto-hipifies `.cu` via
`torch.utils.cpp_extension.load`) and `compile/builders/tvm_ffi_builder.py` (`apache/tvm-ffi` has
first-class HIP). Plan context: `ROCM_PORT_PLAN.md` §3.2, §3.3, §3.11. Agent kernel-gen guidance:
`flashinfer_bench/agents/ffi_prompt.py`.

## Pick a binding

| Binding | Builder | ROCm behavior |
|---|---|---|
| `torch` | `torch_builder` | `torch.utils.cpp_extension.load()` **auto-hipifies** `.cu` at build; treats `with_cuda` as "compile device code". Simplest path. |
| `tvm-ffi` | `tvm_ffi_builder` | tvm-ffi auto-selects `hipcc`, emits `-D__HIP_PLATFORM_AMD__=1 -fno-gpu-rdc`, links `-lamdhip64`, applies `--offload-arch`. |

Keep `.cu` as the source extension (hipcc compiles it). Don't add unconditional `-lcuda -lcublas`;
on ROCm add BLAS **only when needed** as `-lhipblas -lrocblas`.

## Arch targeting

Derive arch from the live device (`gcnArchName`, e.g. `gfx942`) and pass it as a **list** so CDNA4 is
additive, not a rewrite:
- torch builder: `PYTORCH_ROCM_ARCH="gfx942;gfx950"`
- tvm-ffi builder: `TVM_FFI_ROCM_ARCH_LIST="gfx942 gfx950"`
- raw hipcc: `--offload-arch=gfx942`

Don't hand-add `--offload-arch` when a JIT generator already injects it per target arch. See
[`benchmark-on-rocm`](../benchmark-on-rocm/SKILL.md) for arch detection.

## CUDA → ROCm cheat sheet

Most CUDA C++ maps 1:1 (hipify handles `cudaXxx`→`hipXxx`, CUB/Thrust→hipCUB/rocPRIM, `__nv_bfloat16`,
etc. for `.cu`). The non-obvious substitutions when porting a kernel + its PyTorch launcher:

| CUDA / upstream | ROCm / this fork |
|---|---|
| `tvm::ffi::TensorView` (tvm-ffi bindings) | `at::Tensor` (torch bindings) |
| `TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, op)` | `TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m){ m.def("op", op); }` |
| `PYBIND11_MODULE(...)` | **Don't** — use `TORCH_LIBRARY_FRAGMENT` (integrates with `torch.compile`) |
| `TVM_FFI_THROW(ValueError) << "..."` | `TORCH_CHECK(cond, "...")` |
| `get_stream(tensor.device())` | `at::hip::getCurrentHIPStream()` |
| `c10::cuda::OptionalCUDAGuard` | `c10::hip::OptionalHIPGuardMasqueradingAsCUDA` (literally the type name) |
| CUDA event timing | HIP events via `torch.cuda.Event` (already the bench default) |
| NVTX ranges | roctx (maps from `torch.cuda.nvtx.range` on ROCm) |
| `nvcc` extra flags via `extra_cuda_cflags=[...]` | **same kwarg name** — internally routed to `hipcc` |
| warp size 32 | **wavefront = 64** (gfx942 and gfx950) — use `warpSize`, never hardcode 32 |
| FP8 `_fn` (e4m3fn / e5m2) | FP8 **`_fnuz`** (`__hip_fp8_e4m3_fnuz`; torch `float8_e4m3fnuz`) |
| tensor cores / `mma` | MFMA matrix-core intrinsics (via rocWMMA / Composable Kernel) |

If you mirror flashinfer's own HIP launcher style, its `pytorch_extension_utils.h` provides the
validation macros (`CHECK_INPUT`, `CHECK_LAST_DIM_CONTIGUOUS_INPUT`, `CHECK_DIM`, `CHECK_SHAPE`, …)
and dispatch macros (`DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16` = FP16+BF16;
`..._FP8` = E4M3+E5M2, both `_fnuz` on CDNA; unsuffixed `..._TO_CTYPE` = FP16+BF16+FP8). There is no
`_FP16_FP32` variant — dispatch FP32 manually.

## Non-obvious gotchas

- **PyTorch ROCm masquerade.** `input.device.type == "cuda"` even on AMD — never check `"hip"`. HIP
  namespaces are reachable via `at::hip::...` and `c10::hip::OptionalHIPGuardMasqueradingAsCUDA`.
- **Keep device headers framework-agnostic.** No `<torch/...>` includes in `.cuh`/device headers
  (raw pointers only); do the PyTorch glue in the launcher `.cu`. Mixing them causes subtle build
  failures.
- **`-ffast-math` re-enables `-ffinite-math-only`** on clang/hipcc, which breaks kernels that use
  `-inf` as a sentinel (online-softmax). Re-add `-fno-finite-math-only` if you pass fast-math.
  (CUDA's `-use_fast_math` does *not* enable finite-math-only — a real divergence when porting.)
- **`gpu_iface` over duplication.** If a primitive (MMA intrinsic, cross-lane shuffle, dtype
  container, warp reduction) differs between CUDA and HIP, put the HIP variant under a
  `gpu_iface/backend/hip/` header (`mma_hip.h`, `memory_ops_hip.h`, `math_hip.h`, `vec_dtypes_hip.h`)
  and expose a common name — don't fork the whole kernel.
- **CUTLASS is NVIDIA-only.** For matrix-core kernels use Composable Kernel (ck_tile) + rocWMMA, or
  lean on AITER. Tracked in `ROCM_PORT_PLAN.md` §3.7.

## CDNA3 (gfx942) vs CDNA4 (gfx950)

- **Wavefront = 64 on both.** Anything assuming warp = 32 is wrong.
- **FP8** is `_fnuz` on both (`torch.float8_e4m3fnuz`, not `_fn`). Bit-exact parity with NVIDIA FP8
  is not guaranteed — calibrate scale factors separately.
- **MFMA:** CDNA4 has additional FP8 MFMA shapes not on CDNA3. Guard arch-specific intrinsics with
  `__gfx942__` / `__gfx950__` or dispatch at the Python layer.
- **LDS / register / occupancy budgets differ.** Don't hardcode tile sizes — parameterize, or query
  `torch.cuda.get_device_properties(dev)` at plan time.

## Workflow & pre-commit checklist

1. Write the `Solution` with `language: cuda` + chosen `binding`; `.cu` source. Model the entry point
   on existing target solutions and `agents/ffi_prompt.py`.
2. Build+run through the normal loop on the GPU (build → time → correctness vs reference → speedup);
   see [`benchmark-on-rocm`](../benchmark-on-rocm/SKILL.md).
3. On build failure, check arch flags + the cheat-sheet substitutions first; then
   [`debug-rocm`](../debug-rocm/SKILL.md).
4. Clear the stale JIT cache after any toolchain/flag change: `rm -rf ~/.cache/flashinfer/`.

Before committing: no `<torch/...>` in device headers; launcher uses `at::hip::getCurrentHIPStream()`
+ `OptionalHIPGuardMasqueradingAsCUDA`; binding via `TORCH_LIBRARY_FRAGMENT` (not `PYBIND11_MODULE`);
no hardcoded warp=32 or tile sizes; correctness passes vs reference on gfx942.

## Maintaining this document

Update when the torch/tvm-ffi builders change ROCm flag handling (§3.2/3.3), when the CUTLASS→CK
decision (§3.7) lands, or when `agents/ffi_prompt.py` gains HIP-specific guidance (§3.11).

## See Also

- [ROCM_PORT_PLAN.md](../../../ROCM_PORT_PLAN.md) §3.2, §3.3, §3.7, §3.11
- [generate-aiter-solution](../generate-aiter-solution/SKILL.md) — try AITER first
- [benchmark-on-rocm](../benchmark-on-rocm/SKILL.md) — build/time the solution
- [debug-rocm](../debug-rocm/SKILL.md) — runtime fault triage, debug builds
