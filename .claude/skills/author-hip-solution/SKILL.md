---
name: author-hip-solution
description: Write hand-authored C++/HIP bench Solutions that build on ROCm via the torch and tvm-ffi builders. Covers the CUDA→ROCm porting cheat sheet, arch targeting (--offload-arch / PYTORCH_ROCM_ARCH / TVM_FFI_ROCM_ARCH_LIST), HIP stream/guard masquerade, wavefront=64, and FP8 _fnuz. Use when AITER and Triton don't cover an op and you need a native kernel solution on gfx942/gfx950.
---

# Author a HIP Solution

For ops that neither [AITER](../generate-aiter-solution/SKILL.md) nor Triton cover, write a
`language=cuda` bench `Solution` that compiles to a HIP `.so` on ROCm. On AMD the C++ builders
route through **hipcc** and (for tvm-ffi) hipify `.cu` at build time — so a well-written `.cu`
solution builds unchanged in most cases; the work is avoiding CUDA-only assumptions.

Builders: `compile/builders/torch_builder.py` (auto-hipifies `.cu` via
`torch.utils.cpp_extension.load`) and `compile/builders/tvm_ffi_builder.py`. Plan context:
[`ROCM_PORT_PLAN.md`](../../../ROCM_PORT_PLAN.md) §3.2, §3.3, §3.11. Agent kernel-gen guidance:
`flashinfer_bench/agents/ffi_prompt.py`. Run [`rocm-setup`](../rocm-setup/SKILL.md) first.

## Pick a binding

| Binding | Builder | ROCm behavior |
|---|---|---|
| `torch` | `torch_builder` | `torch.utils.cpp_extension.load()` **auto-hipifies** `.cu` at build; interprets `with_cuda` as "compile device code". Simplest path. |
| `tvm-ffi` | `tvm_ffi_builder` | `apache/tvm-ffi` has first-class HIP: auto-selects `hipcc`, emits `-D__HIP_PLATFORM_AMD__=1 -fno-gpu-rdc`, links `-lamdhip64`, applies `--offload-arch`. |

Keep `.cu` as the source extension either way (hipcc compiles it). Do not add unconditional
`-lcuda -lcublas`; on ROCm add BLAS **only when needed** as `-lhipblas -lrocblas`.

## Arch targeting

Derive arch from the live device (`gcnArchName`, e.g. `gfx942`) and pass it as a **list** so CDNA4
is additive:

- torch builder: `PYTORCH_ROCM_ARCH="gfx942;gfx950"`
- tvm-ffi builder: `TVM_FFI_ROCM_ARCH_LIST="gfx942 gfx950"` (tvm-ffi already reads it)
- raw hipcc: `--offload-arch=gfx942`

See [`benchmark-on-rocm`](../benchmark-on-rocm/SKILL.md) for arch detection.

## CUDA → ROCm cheat sheet

Most CUDA C++ maps 1:1; the non-obvious substitutions when porting a kernel/launcher:

| CUDA | ROCm / HIP |
|---|---|
| `cudaXxx` runtime calls | `hipXxx` (hipify handles this for `.cu`) |
| `nvcc`, `libcudart`, `libcuda` | `hipcc`, `libamdhip64` |
| CUB / Thrust | hipCUB / rocPRIM / rocThrust (hipify auto-maps) |
| cuRAND | rocRAND / hipRAND |
| `__nv_bfloat16` etc. | HIP bf16/fp16 types (hipify maps) |
| CUDA event timing | HIP events via `torch.cuda.Event` (already the bench default) |
| current stream (`get_stream()`) | `at::hip::getCurrentHIPStream()` |
| `c10::cuda::OptionalCUDAGuard` | `c10::hip::OptionalHIPGuardMasqueradingAsCUDA` |
| NVTX ranges | roctx (maps from `torch.cuda.nvtx.range` on ROCm) |
| warp size 32 | **wavefront = 64** on gfx942 **and** gfx950 — don't hardcode 32 |
| FP8 `_fn` (e4m3fn/e5m2) | FP8 **`_fnuz`** (`float8_e4m3fnuz` / `float8_e5m2fnuz`) on AMD |
| tensor cores / `mma` | MFMA matrix-core intrinsics (via rocWMMA / Composable Kernel) |

## Gotchas (from the AMD FlashInfer port experience)

- **PyTorch ROCm masquerade:** `tensor.device.type == "cuda"` is `True` on AMD — never branch on
  `"hip"`. HIP guards/streams are named `...MasqueradingAsCUDA`.
- **Don't put framework headers in device headers:** keep `.cuh`/device headers framework-agnostic
  (raw pointers, no `torch`/`at::` includes); do PyTorch glue in the launcher `.cu`.
- **`-ffast-math` re-enables `-fno-finite-math-only`** on hipcc — be deliberate about fast-math if
  you rely on inf/nan behavior.
- **Don't hardcode tile/wave sizes** for one arch; gate on `gcnArchName` where CDNA3/CDNA4 differ
  (MFMA shapes, LDS sizes).
- **CUTLASS is NVIDIA-only.** For matrix-core kernels use Composable Kernel (ck_tile) + rocWMMA, or
  lean on AITER. Tracked in `ROCM_PORT_PLAN.md` §3.7.

## Workflow

1. Write the definition's `Solution` with `language: cuda` and the chosen `binding`; source file(s)
   as `.cu`. Model the entry point on existing target solutions and `agents/ffi_prompt.py`.
2. Build+run through the normal loop on the GPU (build → time → correctness vs reference → speedup);
   see [`benchmark-on-rocm`](../benchmark-on-rocm/SKILL.md).
3. If the build fails, check the arch flags and the cheat-sheet substitutions first; then
   [`debug-rocm`](../debug-rocm/SKILL.md) for runtime faults.
4. Clear the stale JIT cache after toolchain/env changes: `rm -rf ~/.cache/flashinfer/`.

## Maintaining this document

Update when the torch/tvm-ffi builders change their ROCm flag handling (§3.2/3.3), when the CUTLASS→CK
decision (§3.7) lands, or when `agents/ffi_prompt.py` gains HIP-specific guidance (§3.11).

## See Also

- [ROCM_PORT_PLAN.md](../../../ROCM_PORT_PLAN.md) §3.2, §3.3, §3.7, §3.11
- [generate-aiter-solution](../generate-aiter-solution/SKILL.md) — try AITER first
- [benchmark-on-rocm](../benchmark-on-rocm/SKILL.md) — build/time the solution
- [debug-rocm](../debug-rocm/SKILL.md) — runtime fault triage
