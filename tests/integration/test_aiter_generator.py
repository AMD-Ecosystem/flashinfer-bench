"""CPU-only unit tests for the AITER solution generator.

These do not require a GPU or the ``aiter`` package: the generator only emits Python source strings
from a :class:`Definition`, so it is fully testable on CPU / in CI.
"""

from __future__ import annotations

from flashinfer_bench.data import AxisConst, AxisVar, Definition, TensorSpec, TraceSet
from flashinfer_bench.integration.aiter import (
    AITER_OP_TYPES,
    augment_trace_set_with_aiter,
    generate_aiter_solution,
    generate_aiter_solutions,
)

_RMSNORM_REF = "import torch\n\ndef run(x, weight):\n    return x * weight\n"


def _rmsnorm_def(name="rmsnorm_h4096", dtype="float16"):
    return Definition(
        name=name,
        op_type="rmsnorm",
        axes={"M": AxisVar(), "H": AxisConst(value=4096)},
        inputs={
            "x": TensorSpec(shape=["M", "H"], dtype=dtype),
            "weight": TensorSpec(shape=["H"], dtype=dtype),
        },
        outputs={"out": TensorSpec(shape=["M", "H"], dtype=dtype)},
        reference=_RMSNORM_REF,
    )


def _layernorm_def(name="layernorm_h4096", dtype="float16"):
    return Definition(
        name=name,
        op_type="layernorm",
        axes={"M": AxisVar(), "H": AxisConst(value=4096)},
        inputs={
            "x": TensorSpec(shape=["M", "H"], dtype=dtype),
            "weight": TensorSpec(shape=["H"], dtype=dtype),
            "bias": TensorSpec(shape=["H"], dtype=dtype),
        },
        outputs={"out": TensorSpec(shape=["M", "H"], dtype=dtype)},
        reference=_RMSNORM_REF,
    )


def _silu_mul_def(name="silu_mul_h4096", dtype="float16"):
    return Definition(
        name=name,
        op_type="silu_and_mul",
        axes={"M": AxisVar(), "H2": AxisConst(value=8192)},
        inputs={"x": TensorSpec(shape=["M", "H2"], dtype=dtype)},
        outputs={"out": TensorSpec(shape=["M", "H2"], dtype=dtype)},
        reference=_RMSNORM_REF,
    )


def test_op_types_registered():
    for op in ("rmsnorm", "layernorm", "silu_and_mul"):
        assert op in AITER_OP_TYPES


def test_generate_rmsnorm_solution():
    d = _rmsnorm_def()
    sol = generate_aiter_solution(d, eps=1e-5)
    assert sol is not None
    assert sol.definition == "rmsnorm_h4096"
    assert sol.name == "rmsnorm_h4096__aiter"
    assert sol.spec.entry_point == "aiter_rmsnorm.py::run"
    assert sol.spec.language.value == "python"
    assert sol.spec.target_hardware == ["rocm"]
    assert "aiter" in sol.spec.dependencies
    src = sol.sources[0].content
    # Uses the definition's input arg names, calls aiter.rms_norm with the given eps.
    assert "def run(x, weight):" in src
    assert "aiter.rms_norm(x, weight, 1e-05)" in src


def test_generated_solution_arity_matches_definition():
    d = _rmsnorm_def()
    sol = generate_aiter_solution(d)
    # The generated run() must accept exactly the definition's inputs (value-returning style).
    ns: dict = {}
    body = sol.sources[0].content.replace("import aiter", "aiter = None")
    exec(body, ns)  # noqa: S102 - trusted generated source under test
    import inspect

    params = list(inspect.signature(ns["run"]).parameters)
    assert params == list(d.inputs.keys())


def test_generate_layernorm_solution():
    d = _layernorm_def()
    sol = generate_aiter_solution(d)  # default eps -> layernorm 1e-5
    assert sol is not None
    assert sol.spec.entry_point == "aiter_layernorm.py::run"
    src = sol.sources[0].content
    assert "def run(x, weight, bias):" in src
    assert "aiter.layer_norm(x, weight, bias, 1e-05)" in src


def test_generate_silu_and_mul_solution():
    d = _silu_mul_def()
    sol = generate_aiter_solution(d)
    assert sol is not None
    assert sol.spec.entry_point == "aiter_silu_and_mul.py::run"
    src = sol.sources[0].content
    assert "def run(x):" in src
    assert "aiter.silu_and_mul(out, x)" in src
    assert "x.shape[-1] // 2" in src


def test_eps_override_applies():
    assert (
        "aiter.rms_norm(x, weight, 0.001)"
        in generate_aiter_solution(_rmsnorm_def(), eps=1e-3).sources[0].content
    )
    assert (
        "aiter.layer_norm(x, weight, bias, 0.001)"
        in generate_aiter_solution(_layernorm_def(), eps=1e-3).sources[0].content
    )


def test_unsupported_op_type_returns_none():
    d = _rmsnorm_def()
    d = d.model_copy(update={"op_type": "gemm"})
    assert generate_aiter_solution(d) is None


def test_wrong_arity_returns_none():
    d = Definition(
        name="rmsnorm_bad",
        op_type="rmsnorm",
        axes={"M": AxisVar(), "H": AxisConst(value=8)},
        inputs={"x": TensorSpec(shape=["M", "H"], dtype="float16")},  # missing weight
        outputs={"out": TensorSpec(shape=["M", "H"], dtype="float16")},
        reference=_RMSNORM_REF,
    )
    assert generate_aiter_solution(d) is None


def test_unsupported_dtype_returns_none():
    d = _rmsnorm_def(dtype="float32")
    assert generate_aiter_solution(d) is None


def test_generate_many_skips_unsupported():
    defs = {
        "rmsnorm_h4096": _rmsnorm_def(),
        "gemm_x": _rmsnorm_def(name="gemm_x").model_copy(update={"op_type": "gemm"}),
    }
    out = generate_aiter_solutions(defs)
    assert set(out.keys()) == {"rmsnorm_h4096__aiter"}


def test_augment_trace_set_adds_and_dedups():
    defs = {"rmsnorm_h4096": _rmsnorm_def(), "layernorm_h4096": _layernorm_def()}
    ts = TraceSet(root="/tmp/x", definitions=defs, solutions={}, workloads={}, traces={})

    added = augment_trace_set_with_aiter(ts)
    assert added == 2
    names = {s.name for sols in ts.solutions.values() for s in sols}
    assert names == {"rmsnorm_h4096__aiter", "layernorm_h4096__aiter"}
    # each solution is filed under its definition name
    assert ts.solutions["rmsnorm_h4096"][0].definition == "rmsnorm_h4096"

    # Idempotent by default (no duplicates on re-run)
    assert augment_trace_set_with_aiter(ts) == 0
    total = sum(len(v) for v in ts.solutions.values())
    assert total == 2
