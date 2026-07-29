"""AITER integration: generate AITER-backed solutions for supported op-types.

AITER (AMD's ROCm operator library) is a pre-built kernel library. Rather than a compiled builder,
AITER is used here as a *first-class kernel source*: for op-types AITER covers, a thin Python
``Solution`` is generated whose entry point calls the tuned AITER op. Such solutions build with the
PythonBuilder (no compilation) and are benchmarked/verified like any other.

See the generate-aiter-solution skill (.claude/skills/generate-aiter-solution/SKILL.md).
"""

from __future__ import annotations

from .generator import (
    AITER_OP_TYPES,
    augment_trace_set_with_aiter,
    generate_aiter_solution,
    generate_aiter_solutions,
    is_aiter_available,
)

__all__ = [
    "AITER_OP_TYPES",
    "augment_trace_set_with_aiter",
    "generate_aiter_solution",
    "generate_aiter_solutions",
    "is_aiter_available",
]
