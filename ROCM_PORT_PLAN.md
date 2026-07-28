# FlashInfer-Bench → ROCm / AMD CDNA Port Plan

> Status: DRAFT for review. Target: run the full build → benchmark → apply loop on AMD
> Instinct GPUs, CDNA3 (gfx942, MI300X/MI325X) first, CDNA4 (gfx950) and CDNA5 next.

## 1. Decisions (locked)

| Decision | Choice |
|---|---|
| Target scope | **ROCm-only fork** (hard-replace CUDA paths; NVIDIA support not maintained here) |
| First hardware | **CDNA3 / gfx942**; keep CDNA4 (gfx950) / CDNA5 in view (no hard-coding of one arch) |
| Runtime/build env | **`rocm/flashinfer` Docker** (ROCm 7.x + torch-ROCm + flashinfer fork) + AITER on top |
| flashinfer library | **`AMD-Ecosystem/flashinfer`** (`amd-flashinfer`, imported as `flashinfer`) |
| AITER | **First-class kernel source**: auto-generate AITER-backed Python solutions + use as apply/integration backend |
| C++/CUDA kernels | **Hipify now** (torch/tvm-ffi auto-translate) **+ native HIP authoring later** |
| Profiling/debug agents | **Port now**: NCU → rocprofv3 / rocprof-compute (Omniperf); compute-sanitizer → best-effort ROCm equivalents |
| Builder phasing | Python + Triton → Torch (hipify) → TVM-FFI → TileLang |

## 2. Guiding principle: lean on "HIP-as-CUDA"

PyTorch-ROCm exposes AMD GPUs through the **same `torch.cuda.*` API** and the device string
stays **`"cuda"`**. Therefore, as a rule:

- **Do NOT rename `device="cuda"`** or `torch.cuda.*` calls. They already run on HIP.
- Keep the logical taxonomy value `"cuda"` where it means "the GPU device"; only introduce
  new hardware/language tags where we genuinely need to distinguish AMD build toolchains.

This keeps `bench/runner/*`, `bench/evaluators/*`, `_solution_runner.py`, and
`integration/flashinfer/*` almost entirely unchanged. The real work concentrates in
**timing, the C++ builders, dependencies/packaging, profiling agents, and AITER integration**.

## 3. Port map (component by component)

### 3.0 ROCm library mapping (correctness replacements vs performance levers)

Treat ROCm libraries in **two tiers**: (T1) correctness replacements that make the port compile
and run, and (T2) performance levers that also move the benchmark numbers on CDNA3 (and carry to
CDNA4/5). Several NVIDIA deps in this repo have a *tuned* AMD counterpart, not just a 1:1 swap.

| NVIDIA (where in repo) | ROCm replacement | Conversion (T1) | Optimization (T2) |
|---|---|---|---|
| **CUPTI** via `flashinfer.testing` (`bench/timing.py`) | **rocprofiler-sdk / `rocprofv3`** (device-side kernel timing + counters); torch HIP events fallback | Required — CUPTI path gone | **High** — CUPTI-parity device-side timing; same infra feeds profiling agent |
| **cuBLAS** (`-lcublas`, dep mgr) | **hipBLASLt** (preferred) › rocBLAS/hipBLAS | Required for BLAS link | **High** — epilogue fusion + CDNA3 tuning; basis for tuned GEMM baselines |
| **cuDNN** (`cudnn` dep) | **MIOpen** | Replacement | Low (no cuDNN attention path in-repo) |
| **CUTLASS** (submodule, dep mgr, package-data) | **Composable Kernel (CK / ck_tile)** + **rocWMMA** | Required (submodule is NVIDIA) | **High** — CK fused attn/GEMM + rocWMMA MFMA = native-HIP optimization foundation |
| **Nsight Compute `ncu`** (`agents/ncu.py`) | **rocprof-compute (Omniperf)** sections/roofline + **`rocprofv3`** counters | Required (port-now) | **High** — roofline/occupancy/LDS-VGPR analysis is the optimization feedback loop |
| **compute-sanitizer** (`agents/sanitizer.py`) | LLVM **ASAN-for-HIP** + **rocgdb** (memcheck-ish only) | Partial | n/a |
| **NVTX** (`_solution_runner.py`) | **roctx** (roctracer) | Required for profile scoping | — |
| **nvcc / libcudart / libcuda** (builders) | **hipcc / libamdhip64** | Required | — |
| **CUDA events** (proposed timing) | **HIP events** via `torch.cuda.Event` | Works as-is | Zero-dep timing fallback |
| **CUB / Thrust** (in `.cu` solutions) | **hipCUB / rocPRIM / rocThrust** | hipify auto-maps | Neutral→good (rocPRIM tuned) — sampling / DSA-topk |
| **cuRAND** (sampling kernels) | **rocRAND / hipRAND** | Replacement | — |
| **NCCL** (`distribute_kernel.py`, serve) | **RCCL** | Only if multi-GPU | Deferred |
| **flashinfer-python** | **amd-flashinfer** (routes to AITER/CK) | Required | Carries AMD tuning |

**The four libraries that pay off twice** (complete the conversion *and* are the primary CDNA3
optimization surface):

1. **rocprofiler-sdk / rocprofv3** — accurate device-side timing (upgrade §3.1) + counters for the
   profiling agent (§3.10).
2. **hipBLASLt** — tuned/fused GEMM; the specific optimization behind GEMM baselines (§3.9).
   PyTorch-ROCm can also be pointed at the hipBLASLt backend for the reference path.
3. **CK (ck_tile) + rocWMMA** — MFMA matrix-core access for native-HIP kernels; makes the
   "native HIP later" track competitive, not merely correct (§3.7). AITER already bundles CK/ASM,
   so *using* AITER gets most of this for free.
4. **amd-smi (`amdsmi` Python)** — lock/read clocks, power, temp → **deterministic benchmarking**.
   Ties into existing `tools/gpu-lock`, feeds `env_snapshot()`. The original NVIDIA infra has no
   clock-locking, so this makes measurements *better* than the source, not just ported.

### 3.1 Timing — `bench/timing.py`  ⟶ CRITICAL
- **Problem:** uses `flashinfer.testing.bench_gpu_time_with_cupti`. CUPTI is NVIDIA-only and the
  ROCm flashinfer fork does not ship `flashinfer.testing`.
- **Plan:** introduce a small timing-backend abstraction `bench/timing_backends/`:
  - `torch_events` (default, portable): `torch.cuda.Event(enable_timing=True)` start/stop with
    warmup, L2-cache-cold flush (write a large scratch buffer between iters), median over iters.
    HIP events back `torch.cuda.Event` on ROCm, so this "just works".
  - `rocprof` (accurate): device-side kernel durations via the **`rocprofv3` CLI** with structured
    output (rocpd/CSV/JSON), parsed back in Python — the CUPTI-parity path (§3.0 T2).
    **P0-verified:** rocprofiler-sdk's programmatic timing API is **C/C++ only (no Python binding)**,
    so wrap the CLI rather than link the SDK. `torch_events` stays the zero-dep default; `rocprof`
    is opt-in for high-fidelity runs. rocprofv3 ships in ROCm 7.x (our target image).
  - `time_runnable()` keeps its signature; selects backend via env (e.g. `FIB_TIMING_BACKEND`).
  - **Clock locking (measurement quality):** lock/read GPU clocks via **amd-smi (`amdsmi`)** around
    the timed region for deterministic results; integrate with `tools/gpu-lock`. Record locked
    clocks in `env_snapshot()`.
- **Acceptance:** benchmarks produce stable medians on MI300 without any `flashinfer.testing` import.
- **STATUS: DONE (P1).** `bench/timing.py` rewritten to a self-contained `torch.cuda.Event`
  backend (removed the `flashinfer.testing.bench_gpu_time_with_cupti` import — the ROCm build
  exposes an incompatible signature: no `input_args` kwarg). Cold-L2 flush via `FIB_L2_FLUSH_MB`
  (default 256), backend hook via `FIB_TIMING_BACKEND` (rocprof reserved). Validated on gfx942:
  matmul timed at ~0.045 ms; real Benchmark loop reports 15.63x speedup vs reference. Clock-locking
  (amd-smi) and the rocprof backend remain follow-ups.

### 3.2 TVM-FFI builder — `compile/builders/tvm_ffi_builder.py`  ⟶ ~~HIGH RISK~~ LOW-MEDIUM (P0-resolved)
- **Problem:** flashinfer-bench's builder hardcodes CUDA: `_find_cuda_lib_path()` looks for
  `nvcc`/`libcudart.so`; `build()` appends `-lcuda -lcublas`.
- **P0 FINDING (resolved):** upstream `apache/tvm-ffi` **already has first-class HIP support** in
  `tvm_ffi/cpp/extension.py` — the exact `tvm_ffi.cpp.build` API our builder calls. It:
  auto-detects backend via `torch.version.hip` / discoverable `hipcc`; `_find_rocm_home()` honors
  `ROCM_HOME`/`ROCM_PATH`; emits `-D__HIP_PLATFORM_AMD__=1 -fno-gpu-rdc`, links `-lamdhip64`, sets
  the compiler to `hipcc`, and applies `--offload-arch=gfxXXX` (via `TVM_FFI_ROCM_ARCH_LIST` or
  `rocm_agent_enumerator`). `.cu` files are compiled by hipcc on ROCm. So the HIP compile path is
  **provided by the library**, not something we build.
- **Revised plan (much smaller):**
  1. **Remove** flashinfer-bench's CUDA hardcoding: drop `_find_cuda_lib_path()` and the
     unconditional `-lcuda -lcublas`; let tvm-ffi's backend detection choose hipcc + `-lamdhip64`.
  2. Add BLAS libs **only when the solution needs them**, ROCm-flavored: `-lhipblas -lrocblas`.
  3. Wire our arch config (§3.8) to **`TVM_FFI_ROCM_ARCH_LIST`** (tvm-ffi already reads it).
  4. Keep `.cu` as the source extension (hipcc compiles it); optionally accept `.hip` later.
- **Acceptance:** a representative C++/CUDA (default-binding) solution compiles to `.so` and loads
  on gfx942. *(End-to-end run deferred to when the Python stack is installed; toolchain + library
  support already confirmed.)*

### 3.3 Torch builder — `compile/builders/torch_builder.py`
- **Good news:** `torch.utils.cpp_extension.load()` on ROCm **auto-hipifies** `.cu` at build time.
- **Plan:**
  - Keep `with_cuda` (torch-ROCm interprets it as "compile device code"). Verify no CUDA-only
    `extra_cflags` leak in.
  - Rework `DependencyManager._CUDA_DEPS`: `cublas→hipblas/rocblas`, `cudnn→miopen`,
    `cutlass→composable_kernel/hipCUTLASS` (see 3.7). It is currently disabled — re-enable with the
    ROCm map when needed.
  - Add optional `PYTORCH_ROCM_ARCH` / `--offload-arch` targeting from a central arch config (3.8).
- **Acceptance:** an existing `.cu` solution with `binding: torch` builds and runs via hipify.

### 3.4 Triton & TileLang builders — `compile/builders/{triton,tilelang}_builder.py`
- Both are Python-import based (subclass `PythonBuilder`); `is_available()` just checks the import.
- **Plan:** ensure the ROCm-appropriate Triton (from AMD PyPI, ROCm-matched) and TileLang-ROCm are
  installed in the image. No code change expected beyond validation. TileLang is deprioritized.
- **Acceptance:** a Triton kernel solution builds and benchmarks on gfx942.

### 3.5 Python builder & reference path — `compile/builders/python_builder.py`, `registry.py`
- `build_reference()` hardcodes `target_hardware=["cuda"]`. Since references are Python, this does
  not affect building — but for a ROCm-only fork set it to a neutral/`"rocm"` tag for honesty.
- **This is the workhorse for AITER-backed solutions** (see 3.9).

### 3.6 Runners / evaluators — `bench/runner/*`, `bench/evaluators/*`, `_solution_runner.py`
- `torch.cuda.set_device/empty_cache/synchronize`, `mp.get_context("spawn")`, and
  `torch.cuda.nvtx.range` all function on ROCm (nvtx maps to roctx via roctracer).
- **Plan:** mostly leave as-is. Add:
  - Capability/arch detection helper (`gcnArchName`) where SM-capability was implied.
  - Review numeric tolerances in evaluators for fp8/bf16 (AMD vs NVIDIA rounding) — may need
    per-dtype tolerance tuning in `bench/utils.py` / evaluators.
- **Acceptance:** isolated + persistent runners execute solutions on MI300 unchanged.

### 3.7 CUTLASS submodule — `.gitmodules`, `thirdparty/cutlass`, `pyproject` package-data
- **Problem:** submodule is `NVIDIA/cutlass`; `pyproject` ships `flashinfer_bench._deps.cutlass`.
- **Plan:** replace with **Composable Kernel (CK / ck_tile)** + **rocWMMA** as the header
  dependency (see §3.0 T2 — these give MFMA matrix-core access for native-HIP kernels), and/or
  **hipCUTLASS**, OR drop entirely if AITER covers the GEMM/attention needs. Decide during Phase 2.
  Update `.gitmodules`, `tool.setuptools.package-data`, and the dep manager map together.

### 3.8 Arch targeting (CDNA3 → CDNA4/5)
- Add a single source of truth for target arch(s), e.g. `FIB_ROCM_ARCH` (default derived from the
  live device via `torch.cuda.get_device_properties().gcnArchName`, e.g. `gfx942`).
- Thread it into hipcc/torch/tvm-ffi builds as `--offload-arch=`/`PYTORCH_ROCM_ARCH`.
- Keep it a **list** so CDNA4 (`gfx950`) / CDNA5 are additive, not rewrites.

### 3.9 AITER integration (first-class)  ⟶ HIGH VALUE
- AITER is a **pre-built op library** (call it; don't build it). No new builder.
- **Plan:**
  1. Add an `aiter` availability guard + record in `env_snapshot()` and solution `dependencies`.
  2. Add a **preset/generator** that emits AITER-backed **Python solutions** (`language=python`,
     `main.py::run` → `import aiter; aiter.<op>(...)`) for covered op-types: GEMM (incl. fp8/a8w8),
     attention (MHA/MLA/paged/FMHA), fused MoE, RMSNorm, RoPE, quantization. Location:
     alongside `apply/presets.py` / `tracing/presets.py` or a new `integration/aiter/`.
     For **GEMM** specifically, also emit a **hipBLASLt**-backed tuned baseline (§3.0 T2).
     **P0-verified:** hipBLASLt is **C/C++ only (no Python API)** and the standalone repo is retired
     (now in `ROCm/rocm-libraries`; lib still ships as `libhipblaslt`). So consume it **via the
     PyTorch hipBLASLt backend** (torch GEMM with the hipBLASLt backend enabled) or **via AITER's
     tuned GEMM**, not a direct C++ solution initially. gfx942 is supported; **gfx950/CDNA4 coverage
     to confirm** in rocm-libraries.
  3. Wire AITER as a swap-in backend in `integration/flashinfer/adapters/*` (GEMM, attention,
     rmsnorm) for the `apply` path.
  4. Gracefully skip op-types/shapes/dtypes AITER doesn't cover (fall through to Triton/hipified C++).
- **Acceptance:** for ≥1 GEMM and ≥1 attention definition, an AITER solution builds (no compile),
  passes correctness vs reference, and benchmarks.

### 3.10 Profiling / debug agents — `agents/ncu.py`, `agents/sanitizer.py`, `_solution_runner.py`
- **NCU → ROCm:**
  - Replace `ncu` command construction with **`rocprofv3`** (kernel/hip trace, counters) for the
    "raw/details" style output and **`rocprof-compute`** (Omniperf) for detailed sections/sets.
  - Replace `--nvtx --nvtx-include` filtering with **roctx** ranges (`--roctx-trace`); the runner
    already emits a range via `torch.cuda.nvtx.range("flashinfer_bench_ncu_profile")` which maps to
    roctx on ROCm (verify; otherwise call roctx directly).
  - Keep the same tool signature/return-string contract so agent tooling/schema is unchanged.
- **compute-sanitizer → ROCm (best-effort):** no 1:1 equivalent. Provide:
  - `memcheck` → HIP + LLVM AddressSanitizer instrumented build (`-fsanitize=address`, xnack) where
    feasible; and/or `rocgdb`-based checks.
  - `racecheck/synccheck/initcheck` → **document as unsupported** and return a clear message rather
    than failing. Keep the tool callable so agents degrade gracefully.
- **Acceptance:** `flashinfer_bench_run_ncu`-equivalent returns real rocprof output on MI300;
  sanitizer returns memcheck results or an explicit "unsupported on ROCm" per check.

### 3.11 Agent kernel-gen prompt — `agents/ffi_prompt.py`
- Update guidance to target **HIP**: hipcc, HIP headers, `gfx942` specifics (wavefront = 64, LDS
  sizes, matrix-core/MFMA intrinsics, `__launch_bounds__`), and TVM-FFI/HIP entry conventions.
- **Acceptance:** generated C++ solutions compile via the ROCm builders.

### 3.12 Packaging / deps — `pyproject.toml`, `.gitmodules`
- `flashinfer-python>=0.6.4` → **`amd-flashinfer`** (installed from `https://pypi.amd.com/simple/`;
  document the `--index-url`, since it can't be pinned as an index in `pyproject`).
- Replace `cuda12` extra with a `rocm` extra (rocm-matched torch/triton documented via install
  guide, not hard-pinned).
- Keep `apache-tvm-ffi` pending the Phase-0 HIP spike result.
- Document AITER as a source/extra install (not a PyPI pin).

### 3.13 Examples / docs / CI
- `examples/ffi/CMakeLists.txt`: `LANGUAGES CXX CUDA` → `HIP`; `find_package(CUDAToolkit)` →
  `find_package(hip)`; link `hip::device` / `hip::host`. (Low priority — examples.)
- `docs/start/installation.mdx`, `docs/model_coverage.mdx`, `docs/op-types/*`: ROCm install + AITER
  + supported arch matrix.
- CI: current `unit_test.yaml` runs CPU-only on `ubuntu-latest`. Add a **ROCm GPU CI job** on a
  self-hosted MI300 runner (gated by `requires_torch_cuda`) for build/bench smoke tests.

## 4. Milestones

| Phase | Goal | Key deliverables | Exit criteria |
|---|---|---|---|
| **P0 — Env & spikes** | Stand up the platform, de-risk unknowns | rocm/flashinfer Docker + AITER; import smoke tests; **tvm-ffi HIP spike**; nvtx→roctx check | `import flashinfer`, `torch.cuda` on MI300, and a trivial C++ compile all verified; tvm-ffi HIP go/no-go decided |
| **P1 — Core loop** | build→bench→apply green with Python+Triton+AITER | timing backend (3.1); AITER solution generator (3.9); reference/target tag fix (3.5) | GEMM + attention + rmsnorm: reference builds, AITER + Triton solutions pass correctness & benchmark on gfx942 |
| **P2 — C++ via hipify** | Hand-written kernels build | Torch builder deps/arch (3.3); TVM-FFI ROCm toolchain (3.2); CUTLASS→CK (3.7) | ≥1 `.cu` solution each via torch and tvm-ffi bindings builds, runs, matches reference |
| **P3 — Agent tooling** | Profiling/debug on ROCm | rocprof/omniperf NCU replacement + sanitizer best-effort (3.10); HIP FFI prompt (3.11) | profiling tool returns real counters; sanitizer degrades gracefully; agent can generate a compiling HIP kernel |
| **P4 — Packaging/CI/native HIP** | Productionize | pyproject/gitmodules (3.12); ROCm GPU CI (3.13); optional `HIP` language for native authoring; TileLang | fresh Docker build installs clean; GPU CI smoke passes; native-HIP solution path documented |
| **P5 — CDNA4/5 readiness** | Multi-arch | arch-list config validated (3.8) | build+bench pass on gfx950 with only arch-list change |

### P0 results (on-hardware, MI325X / gfx942, ROCm 7.2.0)
Verified directly on the target host **without** installing the heavy Python stack:

| Check | Result | Impact |
|---|---|---|
| HIP compile+run (`hipcc --offload-arch=gfx942`) | ✅ builds & runs | toolchain confirmed on real CDNA3 |
| HIP-event timing (`hipEventElapsedTime`) | ✅ 0.017 ms/iter measured | validates `torch_events` default backend (§3.1) |
| `rocprofv3 --kernel-trace` (CSV) | ✅ per-kernel `Start/End_Timestamp` (+ VGPR/SGPR/LDS/grid) | validates accurate timing backend (§3.1) **and** profiling-agent counters (§3.10) |
| roctx range + `rocprofv3 --marker-trace --kernel-rename` | ✅ `flashinfer_bench_ncu_profile` region captured; kernels renamed to region | validates NCU `--nvtx-include` equivalent (§3.10) |
| tvm-ffi HIP support (source inspection of `apache/tvm-ffi`) | ✅ built-in: `hipcc`, `-lamdhip64`, `--offload-arch`, `TVM_FFI_ROCM_ARCH_LIST` | collapses §3.2 risk to "remove our CUDA hardcoding" |

**Deferred to when the Python stack is installed (validation, not risk):** `import torch`/`torch.cuda`
smoke test, `torch.cuda.nvtx.range`→roctx mapping (C-level roctx already confirmed; runner can call
roctx directly if torch's mapping is absent), and `import flashinfer`/`import aiter`.

**Environment note:** host is bare-metal ROCm 7.2-native (hipcc, rocprofv3, amd-smi, docker, GPU all
present).

### P0 results — Docker stack (8/8 in-container on MI325X / gfx942)
A reproducible ROCm dev image was built and the whole stack validated end-to-end **inside the
container on the GPU**. Files: `docker/rocm/{Dockerfile,run.sh,validate_p0.py,README.md}`.

| Component | Pinned | Source |
|---|---|---|
| Base | `rocm/dev-ubuntu-24.04:7.2-complete` | Docker Hub (anonymous) |
| torch (ROCm) | 2.9.1+rocm7.2.0 | repo.radeon.com rocm-rel-7.2 |
| amd-flashinfer (imports `flashinfer`) | 0.5.3+amd.1 | pypi.amd.com/rocm-7.2.0 |
| amd_aiter | 0.1.10 | pypi.amd.com/rocm-7.1.1 |
| apache-tvm-ffi | 0.1.12 | PyPI |
| git (import-time dep of aiter/flashinfer) | 2.55 | conda-forge via micromamba |

`validate_p0.py` → **8/8 PASS**: torch.cuda on gfx942, GPU matmul, `import flashinfer`/`aiter`/
`flashinfer_bench`, tvm-ffi backend=`hip`, **tvm-ffi HIP kernel compiled+ran on gfx942**, nvtx→roctx.

**Environment gotchas discovered (baked into the Dockerfile):**
- BuildKit ignores the daemon's insecure-registry/mirror config → build with `DOCKER_BUILDKIT=0`.
- A stale Docker Hub credential blocked the anonymous public base pull → `docker logout` first.
- `archive.ubuntu.com` is unreachable and the internal apt mirror lacks git → no apt; install
  cmake/ninja as pip wheels and git from conda-forge.
- Debian system pip 24.0 can't self-uninstall → don't `pip install -U pip`; use `--ignore-installed`
  and `--break-system-packages`.
- aiter/flashinfer call `git` at import (aiter's handler misses `FileNotFoundError`); setuptools_scm
  needs git + `safe.directory '*'` for the bind-mounted repo, and `SETUPTOOLS_SCM_PRETEND_VERSION`.
- flashinfer-bench installed with `pip install -e . --no-deps` so the NVIDIA `flashinfer-python`
  dependency is never pulled (pending the §3.12 pyproject swap).

### P1 progress (branch `rocm/p1-timing-aiter`)
- ✅ **Timing backend** (§3.1): `bench/timing.py` → self-contained `torch.cuda.Event`. Validated on
  gfx942 (matmul ~0.045 ms; Benchmark loop 15.63x speedup vs reference).
- ✅ **Reference target tag** (§3.5): `registry.build_reference` `target_hardware` `cuda`→`rocm`.
- ✅ **End-to-end loop proven on gfx942**: `tests/bench/test_benchmark.py` real Benchmark path —
  persistent runner spawns GPU workers, builds solutions, times them, checks correctness, computes
  speedup (5/6 pass).
- ✅ **AITER approach validated end-to-end** (§3.9): hand-written AITER-backed Python solution
  (`aiter.rms_norm`) for an RMSNorm definition runs through the real Benchmark loop on gfx942 —
  **PASSED, 0.038 ms, 4.32x speedup** vs reference (`docker/rocm/validate_aiter_rmsnorm.py`). This
  proves the pattern the generator will template; auto-generation across op-types is the remaining
  work.
- ✅ **AITER solution generator** (§3.9): `integration/aiter/` with an extensible op-type registry
  (`generate_aiter_solution` / `generate_aiter_solutions`). Handlers for **rmsnorm, layernorm,
  silu_and_mul** (per-op eps defaults; returns None outside AITER's arity/dtype envelope). 10
  CPU-only unit tests. Validated end-to-end on gfx942 (`docker/rocm/validate_aiter_ops.py`):
  **3/3 generator-produced solutions PASSED** — rmsnorm 6.05x, layernorm 4.68x, silu_and_mul 4.32x
  vs reference. Adding attention/MoE/GEMM-variants is now incremental registry work.
- ⏳ Wire AITER solutions into the `apply` path (§3.9 step 3) — next.
- **Pre-existing (non-ROCm) test issues found** — would fail on NVIDIA too, not caused by the port,
  flagged for later: (a) `tests/bench/test_evaluator.py` passes a raw `BenchmarkConfig` (atol=None)
  to `DefaultEvaluator.evaluate` which expects a `ResolvedEvalConfig` (atol default 1e-2) →
  `TypeError` in `compute_error_stats`; (b) `test_benchmark_with_mixed_results` asserts a trace-dump
  path (`traces/op/simple_add.jsonl`) that isn't produced. Both are unrelated to CUDA→ROCm.

## 5. Top risks / open questions
1. **tvm-ffi HIP support** — ✅ RESOLVED (P0): `apache/tvm-ffi` already emits HIP (`hipcc`,
   `-lamdhip64`, `--offload-arch`, `TVM_FFI_ROCM_ARCH_LIST`). Remaining work is removing our
   builder's hardcoded CUDA flags (§3.2) — no custom hipcc path needed.
2. **compute-sanitizer gap** — no full ROCm analog; racecheck/synccheck/initcheck will be
   unsupported. Confirm that's acceptable.
3. **AITER provisioning** — source build, ROCm-version-coupled Triton; must be baked into the image.
4. **Numerical tolerances** — fp8/bf16 rounding differs vs NVIDIA; evaluators may need per-dtype
   tolerance tuning to avoid false correctness failures.
5. **CUTLASS replacement** — CK+rocWMMA vs hipCUTLASS vs drop-and-use-AITER; decide in P2.
6. **rocprofiler-sdk timing API** — ✅ RESOLVED (P0): SDK programmatic API is C/C++ only (no Python
   binding); accurate backend wraps the **`rocprofv3` CLI** and parses structured output. torch HIP
   events remain the default. rocprofv3 present in ROCm 7.x.
7. **hipBLASLt consumption** — ✅ RESOLVED (P0): C/C++ only, standalone repo retired (now in
   `ROCm/rocm-libraries`). Consume via the **PyTorch hipBLASLt backend** or **AITER GEMM**, not a
   direct C++ solution. gfx942 supported; **gfx950/CDNA4 coverage still to confirm**. Fall back to
   rocBLAS/AITER where hipBLASLt lacks a good solution.
8. **Default C++ binding & `target_hardware`** — confirm whether existing target solutions rely on
   the tvm-ffi default and on `target_hardware=["cuda"]` filtering (currently unused in build/verify).

## 6. Explicitly out of scope (this fork)
- Maintaining NVIDIA/CUDA build paths.
- Windows support.
- Full parity of compute-sanitizer race/sync/init checks.
