"""Generate AITER-backed Python solutions from op definitions.

Each op-type handler inspects a :class:`Definition` (inputs/outputs, dtypes) and, if AITER covers it
with matching semantics, returns a :class:`Solution` whose ``run`` calls the tuned AITER op. Handlers
return ``None`` when the definition's shape/dtype/arity is outside AITER's supported envelope, so the
benchmark can fall back to other builders (Triton, hipified C++).

The generated solution is a normal Python solution: it builds via ``PythonBuilder`` (no compilation)
and is verified/benchmarked against the definition's reference like any other.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from flashinfer_bench.data import (
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedLanguages,
)

logger = logging.getLogger(__name__)

# Default RMSNorm epsilon. Definitions embed eps in their reference; we cannot reliably parse it, so
# we use the near-universal 1e-6 and allow override via generate_aiter_solution(..., eps=...).
_DEFAULT_RMSNORM_EPS = 1e-6

# dtypes AITER's fp16/bf16 elementwise/norm ops accept directly.
_FLOAT16_DTYPES = ("float16", "bfloat16")


def is_aiter_available() -> bool:
    """Return True if the ``aiter`` package can be imported."""
    try:
        import aiter  # noqa: F401
    except Exception:
        return False
    return True


def _make_solution(
    definition: Definition, entry_file: str, source: str, *, suffix: str = "aiter"
) -> Solution:
    """Build a Python Solution wrapping an AITER call for ``definition``."""
    return Solution(
        name=f"{definition.name}__{suffix}",
        definition=definition.name,
        author="aiter-generator",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["rocm"],
            entry_point=f"{entry_file}::run",
            dependencies=["aiter"],
            destination_passing_style=False,
        ),
        sources=[SourceFile(path=entry_file, content=source)],
        description=f"AITER-backed {definition.op_type} solution (auto-generated).",
    )


def _input_names(definition: Definition) -> List[str]:
    """Ordered input argument names (dict preserves definition order)."""
    return list(definition.inputs.keys())


def _all_float16ish(definition: Definition) -> bool:
    specs = list(definition.inputs.values()) + list(definition.outputs.values())
    return all(s.dtype in _FLOAT16_DTYPES for s in specs)


# --------------------------------------------------------------------------------------------------
# Per-op-type handlers. Each returns a Solution or None (unsupported for this definition).
# --------------------------------------------------------------------------------------------------


def _gen_rmsnorm(definition: Definition, *, eps: float) -> Optional[Solution]:
    """RMSNorm: out = (x / rms(x)) * weight  ->  aiter.rms_norm(x, weight, eps).

    Requires exactly two inputs (activation, weight), one output, all fp16/bf16.
    """
    args = _input_names(definition)
    if len(args) != 2 or len(definition.outputs) != 1:
        logger.debug("rmsnorm: expected 2 inputs / 1 output, got %d/%d", len(args), len(definition.outputs))
        return None
    if not _all_float16ish(definition):
        logger.debug("rmsnorm: unsupported dtypes for AITER rms_norm")
        return None
    x, weight = args
    source = (
        "import aiter\n\n"
        f"def run({x}, {weight}):\n"
        f"    return aiter.rms_norm({x}, {weight}, {eps!r})\n"
    )
    return _make_solution(definition, "aiter_rmsnorm.py", source)


# op_type -> handler. Handlers accept (definition, eps=...) and ignore kwargs they don't use.
_GENERATORS: Dict[str, Callable[..., Optional[Solution]]] = {
    "rmsnorm": _gen_rmsnorm,
}

# op-types this generator knows how to attempt (may still return None per-definition).
AITER_OP_TYPES = tuple(sorted(_GENERATORS.keys()))


def generate_aiter_solution(
    definition: Definition, *, eps: float = _DEFAULT_RMSNORM_EPS
) -> Optional[Solution]:
    """Generate an AITER-backed Solution for ``definition``, or None if unsupported.

    Parameters
    ----------
    definition : Definition
        The op definition to generate a solution for.
    eps : float
        Epsilon for normalization op-types (rmsnorm). Defaults to 1e-6.

    Returns
    -------
    Optional[Solution]
        A Python solution calling the corresponding AITER op, or None if AITER does not cover this
        op-type / shape / dtype.
    """
    handler = _GENERATORS.get(definition.op_type)
    if handler is None:
        return None
    try:
        return handler(definition, eps=eps)
    except Exception:
        logger.warning("AITER generation failed for '%s'", definition.name, exc_info=True)
        return None


def generate_aiter_solutions(
    definitions: Dict[str, Definition], *, eps: float = _DEFAULT_RMSNORM_EPS
) -> Dict[str, Solution]:
    """Generate AITER solutions for every supported definition in a mapping.

    Returns a dict keyed by generated solution name; definitions AITER does not cover are skipped.
    """
    out: Dict[str, Solution] = {}
    for definition in definitions.values():
        sol = generate_aiter_solution(definition, eps=eps)
        if sol is not None:
            out[sol.name] = sol
    return out
