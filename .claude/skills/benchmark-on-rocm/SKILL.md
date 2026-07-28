---
name: benchmark-on-rocm
description: Time, profile, and validate GPU work on AMD CDNA (gfx942/gfx950) — timing-method selection (torch/HIP events vs rocprofv3 vs omnitrace), rocprofv3 counter presets and roofline, clock pinning and reproducibility, gcnArchName arch recording, fp8/bf16 tolerances, AITER-backend constraints, and CDNA tuning references. Applies to the FlashInfer-Bench Benchmark loop and to any standalone kernel measurement on AMD Instinct.
---

# Benchmark on ROCm

Trustworthy performance measurement on AMD Instinct GPUs. This is ROCm benchmarking know-how first;
where FlashInfer-Bench has a concrete implementation of a step, it's noted as "in this repo."

## Golden rule: correctness before performance

AMD and NVIDIA differ in fp8/bf16 rounding and reduction order. **A "fast" result is meaningless
until it passes correctness against the reference.** When correctness fails *only* on AMD, suspect
tolerances or an fp8 encoding mismatch (below) before blaming the kernel. Always compare a candidate
against the known-good path with `torch.testing.assert_close(rtol=1e-2, atol=1e-2)` for bf16/fp16
*before* trusting any speedup.

## Timing-method matrix

| Method | When | How |
|---|---|---|
| **torch / HIP events** (default) | Any kernel ≳ 50 µs; the zero-dep default | `torch.cuda.Event(enable_timing=True)` → HIP events on ROCm. In this repo, `bench/timing.py` wraps this with warmup, cold-L2 flush, and median-over-iters; selected by `FIB_TIMING_BACKEND=torch_events` (default). |
| **rocprofv3 CLI** | Anything you intend to optimize; device-side accuracy | `rocprofv3 --stats --kernel-trace -- python script.py` for per-kernel `Start/End_Timestamp` (+ VGPR/SGPR/LDS/grid); `-i pmc.txt` for a custom counter set. In this repo this is the reserved `FIB_TIMING_BACKEND=rocprof` backend. |
| **omnitrace / rocprof-compute (Omniperf)** | Host+device timeline; roofline/occupancy sections when Python overhead is suspect | Installed separately; scope with a roctx range. |

CUPTI is NVIDIA-only — there is no CUPTI path on ROCm (the old
`flashinfer.testing.bench_gpu_time_with_cupti` import is gone; `amd-flashinfer` ships an
incompatible signature). `torch.cuda.Event` maps to HIP events, so the default backend just works.

**Cold-L2 flush** matters on CDNA: write a large scratch buffer between iters so each timed call
sees a cold last-level cache, or medians are optimistic and non-comparable. In this repo:
`FIB_L2_FLUSH_MB` (default 256).

## Record arch with every number

Device strings show `cuda:0` on AMD too (HIP-as-CUDA masquerade), so the **unambiguous** markers are:

```python
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, p.gcnArchName, torch.version.hip)   # e.g. MI325X gfx942 7.2.x
```

A `gfx942`/ROCm-7.2 number is **not** comparable to a `gfx950`/ROCm-7.0 number — log all three.
Codenames: MI300X/MI325X = gfx942 = CDNA3; MI355X = gfx950 = CDNA4.

## Reproducibility checklist

1. **Warm up** ≥ 5 iters (10–20 if std is high); the first call includes JIT compile.
2. **Pin clocks** for sub-100-µs kernels:
   ```bash
   rocm-smi --showclocks
   sudo rocm-smi --setperfdeterminism 1900   # or --setsclk / --setmclk to fixed levels
   # ... measure ...
   sudo rocm-smi --resetclocks
   ```
   (`ROCM_PORT_PLAN.md` §3.1 tracks folding `amd-smi` clock lock/read into `tools/gpu-lock` +
   `env_snapshot()` so this becomes automatic.)
3. **Record** `props.name`, `props.gcnArchName`, `torch.version.hip` in the log.
4. **Isolate the GPU:** `HIP_VISIBLE_DEVICES=N` (canonical AMD scoping; `ROCR_VISIBLE_DEVICES` is
   one layer deeper; `CUDA_VISIBLE_DEVICES` is also honored by torch). `tools/gpu-lock` sets this.

## rocprofv3 / Omniperf counter presets

Useful counter groupings when profiling a CDNA kernel:

| Preset | Counters (representative) | Answers |
|---|---|---|
| roofline | `FetchSize`, `WriteSize`, MFMA ops, TCC DRAM requests | compute- or memory-bound? |
| compute | MFMA ops + cycle counters | matrix-core throughput |
| memory | L2 + DRAM breakdown | L2 hit-rate, HBM traffic |
| occupancy | `SQ_WAVES`, `SQ_BUSY_CYCLES`, `SQ_VALU_MFMA_BUSY_CYCLES`, `SQ_INSTS_LDS` | wavefront density |
| stall | `SQ_WAIT_INST_VMEM`, `SQ_WAIT_INST_LDS` | memory-stall diagnosis |

Scope a region with a roctx range (the bench runner emits one via `torch.cuda.nvtx.range`, which maps
to roctx on ROCm) and filter with `rocprofv3 --marker-trace --kernel-rename`. Porting the profiling
agent (`agents/ncu.py`) to rocprofv3 + rocprof-compute is tracked in `ROCM_PORT_PLAN.md` §3.10.

**Troubleshooting:** empty counter CSV usually means the kernel-name regex didn't match the mangled
name — run `rocprofv3 --stats --kernel-trace` first and copy the prefix. Confirm `which rocprofv3`
is on `PATH`. Verify the plain timing path works before adding counters.

## Tolerances (avoid false correctness failures)

AMD rounding differs from NVIDIA; the evaluator's per-dtype tolerances may need tuning
(`ROCM_PORT_PLAN.md` §3.6, risk #4):

| Dtype | atol | rtol | Notes |
|---|---|---|---|
| float32 | 1e-5 | 1e-5 | strict |
| float16 | 1e-3 | 1e-3 | standard |
| bfloat16 | 8e-3 | 1e-2 | 0.8% abs / 1% rel |
| fp8 (`_fnuz` on AMD) | 1e-1 | 2e-1 | prefer a hit-ratio ≥85% check over strict allclose |

AMD fp8 uses the **`_fnuz`** encodings (`float8_e4m3fnuz` / `float8_e5m2fnuz`), which are *not*
bit-compatible with NVIDIA's OCP `_fn` fp8 — a scale-factor/dtype issue, not a tolerance one. See
[`debug-rocm`](../debug-rocm/SKILL.md).

## AITER backend constraints

When a solution/adapter routes attention through AITER (`backend="aiter"`), know the real coverage —
otherwise you'll misread a fallback as a measurement:

- Explicit `backend="aiter"` with `kv_layout != "NHD"` → `ValueError` at plan time.
- Explicit `backend="aiter"` on non-gfx942/gfx950 → `RuntimeError`; `amd-aiter` not importable →
  `ImportError`.
- **Native page sizes** (no flat-gather) are `{128, 256, 1024}` for `amd-aiter >= 0.1.10` (older:
  `{16, 1024}`); other page sizes still run via a flat-gather path (not rejected, but different perf).
- **Auto-selection silently falls back** to the default (fa2/HIP) for: non-NHD layout, custom mask,
  dtype ∉ {fp16,bf16}, `dtype_q != dtype_kv`, `head_dim_qk != head_dim_vo`, non-`NONE` pos-encoding,
  or aiter unimportable. A "no speedup" AITER run is often a silent fallback — verify the backend
  actually engaged. See [`generate-aiter-solution`](../generate-aiter-solution/SKILL.md).

## Running in this repo

```bash
tools/gpu-lock --gpus 1 -- \
  python -m flashinfer_bench run --local <trace_dir> --definitions <def> --save-results

# high-fidelity final pass
FIB_TIMING_BACKEND=rocprof FIB_L2_FLUSH_MB=256 \
  tools/gpu-lock --gpus 1 -- python -m flashinfer_bench run --local <trace_dir> --definitions <def>
```

Dataset-wide checks: [`validate-dataset`](../validate-dataset/SKILL.md) uses this same timing backend.

## CDNA tuning references

When optimizing a CDNA kernel, consult in order:
- **Composable Kernel (CK / ck_tile)** — ground truth for LDS layout, `sched_group_barrier` ratios,
  and tiling on CDNA (read the relevant `qr_ks_vs.hpp` for your hdim/dtype).
- **AITER** — performance reference for fp16 / large-hdim attention on gfx942.
- **HipKittens** (arXiv 2511.08083) — producer/consumer patterns underperform on CDNA; prefer the
  **4-wave interleave** approach.

## Maintaining this document

Update when `bench/timing.py` gains the rocprof backend, when clock-locking lands in
`tools/gpu-lock`/`env_snapshot()`, when evaluator tolerance defaults change, or when AITER coverage
(page sizes / fallback conditions) changes.

## See Also

- [rocm-setup](../rocm-setup/SKILL.md) — environment + essential commands
- [generate-aiter-solution](../generate-aiter-solution/SKILL.md) — AITER coverage reality
- [author-hip-solution](../author-hip-solution/SKILL.md) — arch flags for hand-written kernels
- [debug-rocm](../debug-rocm/SKILL.md) — fp8 / NaN / crash triage
- [ROCM_PORT_PLAN.md](../../../ROCM_PORT_PLAN.md) §3.0–3.1, §3.6, §3.10
