---
name: generate-aiter-solution
description: Generate AITER-backed Python solutions for FlashInfer-Bench definitions on ROCm. AITER is AMD's pre-built tuned op library used as a first-class kernel source — for covered op-types a thin Python Solution (import aiter; aiter.<op>(...)) is emitted, built by the PythonBuilder (no compile), and benchmarked/verified like any other solution. Use to add AITER baselines/solutions for an op-type or to extend generator coverage.
---

# Generate AITER Solution

Turn a `Definition` into an AITER-backed `Solution` on ROCm. AITER is a **pre-built op library** —
call it, don't build it — so these solutions carry AMD's tuning for free and run through the normal
Benchmark loop.

Implementation: `flashinfer_bench/integration/aiter/` (`generator.py`). Design:
[`ROCM_PORT_PLAN.md`](../../../ROCM_PORT_PLAN.md) §3.9. Proven end-to-end by
`docker/rocm/validate_aiter_ops.py`. Run [`rocm-setup`](../rocm-setup/SKILL.md) first.

## The public API

```python
from flashinfer_bench.integration.aiter import (
    AITER_OP_TYPES,          # tuple of op_types the generator currently covers
    is_aiter_available,      # bool — is `aiter` importable on this host?
    generate_aiter_solution, # (definition, *, eps=None) -> Optional[Solution]
    generate_aiter_solutions,# (dict[name, Definition], *, eps=None) -> dict[name, Solution]
)
```

- `generate_aiter_solution(definition, eps=None)` returns a Python `Solution` whose entry point
  calls the matching AITER op, or **`None`** if AITER doesn't cover that op-type / shape / dtype
  (graceful fallthrough — the caller then tries Triton or a hipified C++ solution).
- `generate_aiter_solutions(defs)` maps over a dict and skips the uncovered ones.
- `eps` sets the epsilon for normalization ops (defaults: rmsnorm `1e-6`, layernorm `1e-5`).

Covered op-types are whatever `AITER_OP_TYPES` reports (currently `rmsnorm`, `layernorm`,
`silu_and_mul`); this grows as generators are added. **Always read `AITER_OP_TYPES` at runtime**
rather than hard-coding the list.

## Usage

### Generate for a dataset definition

```python
import json
from pathlib import Path
from flashinfer_bench.data import Definition
from flashinfer_bench.integration.aiter import generate_aiter_solution, is_aiter_available

assert is_aiter_available(), "aiter not importable — run rocm-setup"

d = Definition(**json.loads(Path("tmp/flashinfer-trace/definitions/rmsnorm/rmsnorm_h4096.json").read_text()))
sol = generate_aiter_solution(d)            # None if unsupported
if sol is None:
    print("AITER does not cover this definition; fall back to Triton/HIP")
```

The emitted solution is `language=python` and looks like the proven pattern in
`docker/rocm/validate_aiter_ops.py`:

```python
import aiter

def run(x, weight):
    return aiter.rms_norm(x, weight, 1e-6)
```

### Benchmark it

Feed the generated solution into the normal loop (build → time → correctness vs reference →
speedup). `validate_aiter_ops.py` shows the in-memory `TraceSet` path; for dataset solutions, add
the solution under `solutions/` and run `flashinfer-bench run` (see
[`benchmark-on-rocm`](../benchmark-on-rocm/SKILL.md)).

## Extending coverage (adding a new op-type generator)

Each op-type has a small `_gen_<op>(definition, *, eps)` handler registered in `_GENERATORS` inside
`generator.py`. To add one:

1. **Confirm AITER has the op** and its exact signature: `python -c "import aiter; help(aiter.<op>)"`.
   Mind AITER's constraints (dtype/layout/shape coverage) — it does not cover everything.
2. **Write `_gen_<op>`**: validate the definition's inputs/axes/dtypes match what AITER accepts
   (reuse helpers like `_input_names`, `_all_float16ish`); return `None` when they don't rather than
   emitting a solution that will fail at runtime.
3. **Emit the solution** via `_make_solution(...)` with a `run(...)` body whose arg order matches the
   definition's inputs (there's a test that asserts arity/order — keep it green).
4. **Register** it in `_GENERATORS`; it automatically appears in `AITER_OP_TYPES`.
5. **Test**: extend `tests/integration/test_aiter_generator.py` (registration + a shape it should
   accept + a shape/dtype it should reject → `None`), and add the op-type to
   `docker/rocm/validate_aiter_ops.py` for an on-GPU end-to-end check.

Keep generators **pure and import-light** — they must not `import aiter` at module import (only the
generated solution imports it at run time), so definition generation works on hosts without AITER.

## Fallthrough contract

A `None` return is normal and expected. Callers must handle it: try Triton, then a hipified C++
solution ([`author-hip-solution`](../author-hip-solution/SKILL.md)). Never treat "AITER didn't cover
it" as an error.

## Maintaining this document

Update when the `integration/aiter` public API changes, when new op-type generators land (so the
covered list here matches `AITER_OP_TYPES`), or when the validation script is renamed. Do not
hard-code the op-type list — point at `AITER_OP_TYPES`.

## See Also

- `flashinfer_bench/integration/aiter/generator.py` — implementation
- `docker/rocm/validate_aiter_ops.py` — end-to-end proof + template
- [ROCM_PORT_PLAN.md](../../../ROCM_PORT_PLAN.md) §3.9 — AITER integration plan
- [benchmark-on-rocm](../benchmark-on-rocm/SKILL.md) — run the generated solution
- [rocm-setup](../rocm-setup/SKILL.md) — install AITER
