---
name: rocm-debug
description: Debug HIP kernel crashes, memory-access faults, NaN/Inf, HIP OOM, FP8 dtype mismatches, and AITER errors on AMD CDNA (gfx942/gfx950). Covers the AMD_SERIALIZE_KERNEL + HIP_LAUNCH_BLOCKING combo, a per-error recipe table, the rocgdb / AMD_LOG_LEVEL / HSA / dmesg tooling set, AMD gotchas (ROCm masquerade, wavefront=64, fnuz FP8, stripped installs), and why compute-sanitizer has no ROCm analog. Use when a build or benchmark run crashes or produces wrong numbers on ROCm.
---

# Debug ROCm

HIP crash and correctness triage on AMD. This is ROCm debugging know-how; it applies to bench
solution runs, standalone repros, and hand-written HIP kernels alike.

## The magic env-var combo

HIP launches are async, so a Python traceback points at the wrong line. For any unknown fault, set
these **before** running so the report pins to the actual faulting kernel:

```bash
export AMD_SERIALIZE_KERNEL=3   # wait before AND after each kernel — localizes the fault
export HIP_LAUNCH_BLOCKING=1    # synchronous launches — tracebacks land on the right call
```

Both are near-zero-overhead; leave them on while iterating on a new kernel. For an in-script view of
what's being passed, print `t.shape, t.dtype, t.device, t.is_contiguous()` and call
`torch.cuda.synchronize()` immediately before the suspect op.

## Per-error recipe

| Symptom | First check |
|---|---|
| `Memory access fault by GPU node-N` / `hipErrorIllegalAddress` / "CUDA error: illegal memory access" (torch-ROCm reports HIP errors as "CUDA") | Run with the env combo above; print shapes/dtypes/strides just before the call. Verify `is_contiguous()` where required, all tensors on the same `cuda:N`, index tensors (`kv_indices`) within `[0, num_pages)`, and matching `head_dim` between Q and KV. Then `sudo dmesg -T | tail -50`. |
| NaN / Inf in outputs | Wrap with `torch.isnan(t).any()` / `torch.isinf(t).any()`. Causes on CDNA: `_fnuz` FP8 has a different representable range than NVIDIA OCP FP8, so scales calibrated on NVIDIA refs overflow; `-inf` sentinel from a prior op fed into `exp`; or `torch.empty` used where `torch.zeros` was needed. |
| `HIP out of memory` | `rocm-smi --showmeminfo vram --showpids` → kill zombies. JIT-compile spike → `MAX_JOBS=1`. Other tenant → `HIP_VISIBLE_DEVICES=N`. Reduce workload/batch size. |
| `expected scalar type X but found Y` at an FP8 callsite | torch dtype for AMD FP8 is `torch.float8_e4m3fnuz` / `torch.float8_e5m2fnuz`, **not** `torch.float8_e4m3fn` (NVIDIA OCP). A callsite expecting `_fn` mis-dispatches on ROCm. |
| `backend="aiter"` `ValueError` before launch | `kv_layout != "NHD"` — only NHD is allowed (raised in the prefill wrapper's `plan()`). |
| `backend="aiter"` `RuntimeError` | non-gfx942/gfx950 GPU. |
| `backend="aiter"` `ImportError` | `amd-aiter` not installed → `pip install amd-aiter --index-url https://pypi.amd.com/simple/` (or the version pinned by `docker/rocm/`). AITER also calls `git` at import — ensure git present + `git config --global --add safe.directory '*'`. |
| `backend="aiter"` hard GPU fault mid-kernel | `amd-aiter` version mismatch vs ROCm. Reinstall matching your ROCm version; run the default HIP backend to confirm the bug is in AITER, not the caller. |
| Build fails citing `nvcc` / `libcuda` | a CUDA-only flag leaked into a builder — drop unconditional `-lcuda -lcublas`, let hipcc/tvm-ffi choose HIP libs. See [`add-rocm-kernel`](../add-rocm-kernel/SKILL.md). |
| Stale build after an env/flag change | JIT `build.ninja` is only (re)written when missing — env changes are silent no-ops. Clear the cache: `rm -rf ~/.cache/flashinfer/`. |

## ROCm tooling

| Tool | Use |
|---|---|
| `rocgdb --args python my_script.py` | cuda-gdb equivalent. Inside: `catch throw`, `run`, `bt`, `info agents`, `info wavefronts`. |
| `ROCM_DEBUG_WAIT_FOR_DEBUGGER=1` | process blocks at first GPU API call until `rocgdb -p <pid>` attaches. |
| `AMD_LOG_LEVEL=3` (or `4`) | HIP API + stream trace. Linear under `HIP_LAUNCH_BLOCKING=1`, so each Python call maps 1:1 to HIP launches. |
| `HSA_ENABLE_DEBUG=1` | HSA-layer trace (queues, agents — one below HIP). |
| `sudo dmesg -T \| grep -iE 'amdgpu\|kfd\|vm_fault'` | `VM_CONTEXT1_PROTECTION_FAULT_STATUS` gives page-fault class, access type, and offending address when Python only says `hipErrorIllegalAddress`. |
| `watch -n 1 'rocm-smi --showuse --showmeminfo vram --showpids'` | hang diagnosis: 100% GPU + no SQ activity = looping kernel; VRAM pinned after exit = another process holds it. |
| `FLASHINFER_JIT_VERBOSE=1` | verbose JIT build output (amd-flashinfer). |

`compute-sanitizer` / `cuda-gdb` have **no direct ROCm equivalent.** The closest workflow is the
env-var combo + `rocgdb`; `memcheck` is partially covered by an LLVM-ASan-instrumented HIP build
(`-fsanitize=address`). `racecheck`/`synccheck`/`initcheck` are unsupported — the bench sanitizer
agent degrades gracefully and returns "unsupported on ROCm".

## AMD gotchas

- **PyTorch ROCm masquerade.** Device strings show `cuda:0` on AMD; "CUDA error" messages may be
  HIP errors. `tensor.device.type == "cuda"` is `True` on AMD — never test for `"hip"`. The
  unambiguous arch field is `torch.cuda.get_device_properties(0).gcnArchName`.
- **Wavefront = 64**, not 32 — on both gfx942 and gfx950. A representative-thread `printf` ported
  from CUDA needs `threadIdx.x % 64 == 0` (or use the `warpSize` builtin).
- **HIP installs are stripped.** `rocgdb` reports `no symbol table loaded` unless you rebuild the op
  with `-g` (append `"-O0","-g"` via the solution's `extra_cuda_cflags`, then clear
  `~/.cache/flashinfer/`). A generic `FLASHINFER_JIT_DEBUG`-style flag does **not** add debug flags
  on the HIP path.
- **Device `printf` flushes on `torch.cuda.synchronize()`** — same as CUDA.
- **`HIP_VISIBLE_DEVICES`** is the canonical AMD scoping var (`ROCR_VISIBLE_DEVICES` one layer
  deeper); torch also honors `CUDA_VISIBLE_DEVICES`.

## Quick recipes

```bash
# Hard GPU fault — localize it
export AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1
python my_script.py          # traceback now points at the faulting call
sudo dmesg -T | tail -50     # page-fault class / address

# Step into a kernel
export AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1
rocgdb --args python my_script.py
# (rocgdb) catch throw ; run ; bt ; info wavefronts

# HIP API trace
AMD_LOG_LEVEL=3 HIP_LAUNCH_BLOCKING=1 python my_script.py 2> hip.trace
# grep hipLaunchKernel / hipMemcpy / error in hip.trace
```

## Maintaining this document

Update when new AITER error modes appear, when the sanitizer agent (`agents/sanitizer.py`) gains
deeper ROCm coverage (LLVM ASan / rocgdb), or when JIT/cache debug behavior changes.

## See Also

- [rocm-setup](../rocm-setup/SKILL.md) — validate the environment first (`validate_p0.py`)
- [rocm-benchmark](../rocm-benchmark/SKILL.md) — tolerances, AITER fallback conditions
- [add-rocm-kernel](../add-rocm-kernel/SKILL.md) — build-side faults, debug builds
- [generate-aiter-solution](../generate-aiter-solution/SKILL.md) — AITER coverage/fallthrough
