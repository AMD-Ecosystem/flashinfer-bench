---
name: benchmark-on-rocm
description: Time and profile FlashInfer-Bench solutions on AMD CDNA GPUs — choose the timing backend (portable torch-event default vs accurate rocprofv3), pin clocks for stable medians, detect the target arch via gcnArchName, and set correct fp8/bf16 tolerances so AMD-vs-NVIDIA rounding doesn't cause false correctness failures. Use when running the Benchmark loop or interpreting perf numbers on gfx942/gfx950.
---

# Benchmark on ROCm

How the benchmark loop times and validates solutions on AMD, and how to get trustworthy numbers.

The timing implementation is `flashinfer_bench/bench/timing.py`; the design and remaining work are
in [`ROCM_PORT_PLAN.md`](../../../ROCM_PORT_PLAN.md) §3.1 and §3.6. Run
[`rocm-setup`](../rocm-setup/SKILL.md) first.

## Golden rule: correctness before performance

AMD and NVIDIA differ in fp8/bf16 rounding and in reduction order. **Never trust a speedup until the
solution passes correctness against the reference.** A "fast" kernel that fails the evaluator is not
a result. When correctness fails only on AMD, suspect tolerances (below), not the kernel.

## Timing backends

Selected via `FIB_TIMING_BACKEND`; `time_runnable()` keeps its signature regardless.

| Backend | Env | What it does | When |
|---|---|---|---|
| `torch_events` (default) | *(unset)* or `FIB_TIMING_BACKEND=torch_events` | `torch.cuda.Event(enable_timing=True)` start/stop — HIP events on ROCm — with warmup, cold-L2 flush, median over iters. Zero extra deps. | Default for all runs |
| `rocprof` (reserved) | `FIB_TIMING_BACKEND=rocprof` | Device-side kernel durations via the **`rocprofv3` CLI** (structured CSV/JSON parsed back in Python). CUPTI-parity accuracy. | High-fidelity runs; feeds the profiling agent |

Notes:
- The NVIDIA-only `flashinfer.testing.bench_gpu_time_with_cupti` path is **gone** — `amd-flashinfer`
  ships an incompatible signature and CUPTI is NVIDIA-only.
- `torch.cuda.Event` maps to HIP events on ROCm, so the default backend "just works" on AMD.
- **Cold-L2 flush**: `FIB_L2_FLUSH_MB` (default 256 MiB) writes a scratch buffer between iters so
  each timed call sees a cold last-level cache — critical for stable, comparable medians on CDNA.

## Deterministic measurement (clock pinning)

CDNA clocks drift with temperature/power, adding variance. For reproducible medians, pin clocks
around the timed region and record them:

```bash
rocm-smi --setperfdeterminism 1900        # or: rocm-smi --setsclk / --setmclk to fixed levels
rocm-smi --showclocks                      # capture into the run's env snapshot
# ... run the benchmark ...
rocm-smi --resetclocks
```

`ROCM_PORT_PLAN.md` §3.1 tracks integrating `amd-smi` (`amdsmi` Python) clock lock/read into
`tools/gpu-lock` and `env_snapshot()` so this is automatic — until then, pin manually for
apples-to-apples comparisons. Isolate the device with `HIP_VISIBLE_DEVICES` (works like
`CUDA_VISIBLE_DEVICES`; both are honored on ROCm).

## Arch detection

Use `gcnArchName` as the unambiguous marker — do **not** infer from device name strings:

```python
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, p.gcnArchName)   # e.g. "AMD Instinct MI325X" "gfx942"
```

Thread the target arch into builds via `--offload-arch` / `PYTORCH_ROCM_ARCH` /
`TVM_FFI_ROCM_ARCH_LIST` (see [`author-hip-solution`](../author-hip-solution/SKILL.md)). Keep it a
**list** so gfx950/CDNA4 is additive. gfx942 and gfx950 both have wavefront = 64.

## Tolerances (avoid false correctness failures)

AMD rounding differs from NVIDIA; the evaluator's tolerances may need per-dtype tuning
(`ROCM_PORT_PLAN.md` §3.6, risk #4). Sensible starting points:

| Dtype | atol | rtol | Notes |
|---|---|---|---|
| float32 | 1e-5 | 1e-5 | strict |
| float16 | 1e-3 | 1e-3 | standard |
| bfloat16 | 8e-3 | 1e-2 | 0.8% abs / 1% rel |
| fp8 (`_fnuz` on AMD) | 1e-1 | 2e-1 | prefer a hit-ratio ≥85% check over strict allclose |

Note AMD fp8 uses the `_fnuz` variants (`float8_e4m3fnuz` / `float8_e5m2fnuz`), which differ from
NVIDIA's `_fn` encodings — a dtype mismatch, not a tolerance issue. See
[`debug-rocm`](../debug-rocm/SKILL.md).

## Running

```bash
# Under gpu-lock so HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES is set for you
tools/gpu-lock --gpus 1 -- \
  python -m flashinfer_bench run --local <trace_dir> --definitions <def> --save-results

# Force the accurate backend for a final measurement pass
FIB_TIMING_BACKEND=rocprof FIB_L2_FLUSH_MB=256 \
  tools/gpu-lock --gpus 1 -- python -m flashinfer_bench run --local <trace_dir> --definitions <def>
```

For dataset-wide checks use [`validate-dataset`](../validate-dataset/SKILL.md) (its GPU checks use
this same timing backend).

## Profiling (deeper analysis)

Beyond wall-clock, `rocprofv3` gives per-kernel timing + counters (VGPR/SGPR/LDS/grid), and
`rocprof-compute` (Omniperf) gives roofline/occupancy sections — the ROCm replacements for
`ncu`. Scope a region with a roctx range (the runner emits one that maps from
`torch.cuda.nvtx.range`). Porting the profiling agent (`agents/ncu.py`) to these is tracked in
`ROCM_PORT_PLAN.md` §3.10.

## Maintaining this document

Update when `bench/timing.py` gains the `rocprof` backend, when clock-locking lands in
`tools/gpu-lock`/`env_snapshot()`, or when evaluator tolerance defaults change.

## See Also

- [rocm-setup](../rocm-setup/SKILL.md) — environment prerequisite
- [author-hip-solution](../author-hip-solution/SKILL.md) — arch flags for hand-written kernels
- [debug-rocm](../debug-rocm/SKILL.md) — fp8/NaN/crash triage
- [ROCM_PORT_PLAN.md](../../../ROCM_PORT_PLAN.md) §3.0–3.1, §3.6, §3.10
