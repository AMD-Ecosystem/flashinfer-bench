---
name: generate-aiter-solution
description: Use AITER (AMD's pre-built tuned ROCm op library) as a first-class kernel source for FlashInfer-Bench. Generate thin Python Solutions (import aiter; aiter.<op>(...)) that build with the PythonBuilder (no compile) and benchmark/verify like any other solution, know AITER's real coverage and fallback conditions, and extend the generator to new op-types. Use to add AITER solutions/baselines or to reason about when AITER applies on gfx942/gfx950.
---

# Generate AITER Solution

AITER is a **pre-built op library** — call it, don't build it. For op-types AITER covers, a thin
Python `Solution` (`import aiter; aiter.<op>(...)`) carries AMD's tuning for free and runs through the
normal Benchmark loop (build with the PythonBuilder → time → correctness vs reference → speedup).

This skill is AITER-as-kernel-source know-how; the implementation in this repo is
`flashinfer_bench/integration/aiter/` and the end-to-end proof is
`docker/rocm/validate_aiter_ops.py` (design: `ROCM_PORT_PLAN.md` §3.9).

## AITER coverage reality (read before generating)

AITER does **not** cover everything, and misuse fails in specific ways — knowing this up front saves
a misread benchmark:

- **Availability:** `amd-aiter` importable only on ROCm; on non-gfx942/gfx950 an explicit AITER path
  raises `RuntimeError`; unimportable → `ImportError`. AITER also calls `git` at import (ensure git +
  `git config --global --add safe.directory '*'`). Check with `is_aiter_available()` /
  `flashinfer.aiter_utils.is_aiter_supported`.
- **Attention (`backend="aiter"`) constraints:** `kv_layout != "NHD"` → `ValueError` at plan time;
  **native page sizes** (no flat-gather) are `{128, 256, 1024}` for `amd-aiter >= 0.1.10` (older:
  `{16, 1024}`), others take a slower flat-gather path; **auto-select silently falls back** to
  fa2/HIP for non-NHD, custom mask, dtype ∉ {fp16,bf16}, `dtype_q != dtype_kv`,
  `head_dim_qk != head_dim_vo`, or non-`NONE` pos-encoding. A "no speedup" AITER run is often a
  silent fallback — confirm the backend actually engaged. (See
  [`rocm-benchmark`](../rocm-benchmark/SKILL.md) and [`rocm-debug`](../rocm-debug/SKILL.md).)
- **Op coverage varies by version.** Norm/silu/rope-style elementwise ops are broadly available;
  attention is opt-in and constrained as above. Always treat an uncovered op/shape/dtype as a normal
  fallthrough, not an error.

## The generator API (in this repo)

```python
from flashinfer_bench.integration.aiter import (
    AITER_OP_TYPES,          # tuple of op_types currently covered — read at runtime, don't hard-code
    is_aiter_available,      # bool
    generate_aiter_solution, # (definition, *, eps=None) -> Optional[Solution]
    generate_aiter_solutions,# (dict[name, Definition], *, eps=None) -> dict[name, Solution]
)
```

`generate_aiter_solution` returns a Python `Solution` calling the matching AITER op, or **`None`**
when AITER doesn't cover that op-type/shape/dtype (graceful fallthrough). `eps` sets normalization
epsilon (defaults: rmsnorm `1e-6`, layernorm `1e-5`).

## Usage

```python
import json
from pathlib import Path
from flashinfer_bench.data import Definition
from flashinfer_bench.integration.aiter import generate_aiter_solution, is_aiter_available

assert is_aiter_available(), "aiter not importable — see rocm-setup"
d = Definition(**json.loads(Path(".../definitions/rmsnorm/rmsnorm_h4096.json").read_text()))
sol = generate_aiter_solution(d)          # None -> fall back to Triton / hand-written HIP
```

The emitted solution is `language=python`, e.g.:

```python
import aiter

def run(x, weight):
    return aiter.rms_norm(x, weight, 1e-6)
```

Feed it into the normal loop (`docker/rocm/validate_aiter_ops.py` shows the in-memory `TraceSet`
path; for dataset solutions, add it under `solutions/` and run `flashinfer-bench run`). Timing /
tolerances: [`rocm-benchmark`](../rocm-benchmark/SKILL.md).

## Extending coverage (new op-type generator)

Each op-type has a small `_gen_<op>(definition, *, eps)` handler registered in `_GENERATORS` in
`generator.py`. To add one:

1. **Confirm the AITER op + signature:** `python -c "import aiter; help(aiter.<op>)"`; note its
   dtype/layout/shape constraints.
2. **Write `_gen_<op>`:** validate the definition's inputs/axes/dtypes against what AITER accepts
   (reuse `_input_names`, `_all_float16ish`); **return `None`** when they don't, rather than emitting
   a solution that fails at runtime.
3. **Emit** via `_make_solution(...)` with a `run(...)` whose arg order matches the definition's
   inputs (a test asserts arity/order — keep it green).
4. **Register** in `_GENERATORS`; it auto-appears in `AITER_OP_TYPES`.
5. **Test:** extend `tests/integration/test_aiter_generator.py` (registration + an accepted shape +
   a rejected shape/dtype → `None`) and add the op-type to `docker/rocm/validate_aiter_ops.py` for an
   on-GPU end-to-end check.

Keep generators **import-light** — do not `import aiter` at module import (only the *generated*
solution imports it at run time), so definition generation works on hosts without AITER.

## Maintaining this document

Update when the `integration/aiter` public API changes, when new op-type generators land (keep the
coverage claims pointing at `AITER_OP_TYPES`, not a hard-coded list), or when AITER's version-gated
coverage (page sizes / fallback conditions) changes.

## See Also

- `flashinfer_bench/integration/aiter/generator.py` — implementation
- `docker/rocm/validate_aiter_ops.py` — end-to-end proof + template
- [ROCM_PORT_PLAN.md](../../../ROCM_PORT_PLAN.md) §3.9 — AITER integration plan
- [rocm-benchmark](../rocm-benchmark/SKILL.md) — verify the backend engaged; tolerances
- [rocm-debug](../rocm-debug/SKILL.md) — AITER error modes
- [rocm-setup](../rocm-setup/SKILL.md) — install AITER
