"""Tests for the ROCm best-effort sanitizer agent API.

Self-contained (no external FIB_DATASET_PATH dependency): builds an in-memory trace set with a
destination-passing RMSNorm solution. Verifies memcheck runs on ROCm and that the sub-tools with
no ROCm equivalent (racecheck/initcheck/synccheck) degrade gracefully.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from flashinfer_bench.agents.sanitizer import flashinfer_bench_run_sanitizer
from flashinfer_bench.data import (
    AxisConst,
    AxisVar,
    BuildSpec,
    Definition,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    save_json_file,
)
from flashinfer_bench.data.workload import RandomInput, Workload

H = 4096
_DPS_SRC = (
    "import torch\n\n"
    "def run(x, weight, out):\n"
    "    xf = x.float()\n"
    "    var = xf.pow(2).mean(-1, keepdim=True)\n"
    "    out.copy_((xf * torch.rsqrt(var + 1e-6)).to(x.dtype) * weight)\n"
)


def _make_dataset(tmp: Path):
    definition = Definition(
        name="rmsnorm_h4096",
        op_type="rmsnorm",
        axes={"M": AxisVar(), "H": AxisConst(value=H)},
        inputs={
            "x": TensorSpec(shape=["M", "H"], dtype="float16"),
            "weight": TensorSpec(shape=["H"], dtype="float16"),
        },
        outputs={"out": TensorSpec(shape=["M", "H"], dtype="float16")},
        reference="import torch\n\ndef run(x, weight):\n    return x * weight\n",
    )
    save_json_file(definition, tmp / "definitions" / "rmsnorm_h4096.json")
    solution = Solution(
        name="rmsnorm_dps",
        definition="rmsnorm_h4096",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["rocm"],
            entry_point="impl.py::run",
            destination_passing_style=True,
        ),
        sources=[SourceFile(path="impl.py", content=_DPS_SRC)],
    )
    workload = Workload(
        axes={"M": 1024}, inputs={"x": RandomInput(), "weight": RandomInput()}, uuid="wl"
    )
    return solution, workload


def test_run_sanitizer_invalid_type():
    with tempfile.TemporaryDirectory() as d:
        solution, workload = _make_dataset(Path(d))
        out = flashinfer_bench_run_sanitizer(
            solution, workload, trace_set_path=d, sanitizer_types=["bogus"]
        )
    assert out.startswith("ERROR:")
    assert "Invalid sanitizer type" in out


def test_run_sanitizer_unsupported_types_degrade():
    """racecheck/initcheck/synccheck have no ROCm equivalent -> reported, not errored."""
    with tempfile.TemporaryDirectory() as d:
        solution, workload = _make_dataset(Path(d))
        out = flashinfer_bench_run_sanitizer(
            solution,
            workload,
            trace_set_path=d,
            sanitizer_types=["racecheck", "initcheck", "synccheck"],
        )
    assert not out.startswith("ERROR:")
    for tool in ("racecheck", "initcheck", "synccheck"):
        assert f"{tool} is not supported on ROCm" in out


def test_run_sanitizer_invalid_definition():
    with tempfile.TemporaryDirectory() as d:
        solution, workload = _make_dataset(Path(d))
        bad = Solution(
            name="fake",
            definition="nonexistent_def_xyz",
            author="test",
            spec=solution.spec,
            sources=solution.sources,
        )
        out = flashinfer_bench_run_sanitizer(bad, workload, trace_set_path=d)
    assert out.startswith("ERROR:")
    assert "not found" in out


@pytest.mark.requires_torch_cuda
def test_run_sanitizer_memcheck_runs():
    with tempfile.TemporaryDirectory() as d:
        solution, workload = _make_dataset(Path(d))
        out = flashinfer_bench_run_sanitizer(
            solution, workload, device="cuda:0", trace_set_path=d, sanitizer_types=["memcheck"]
        )
    assert not out.startswith("ERROR:")
    # A correct kernel should not trip the fault detector.
    assert "MEMCHECK: no GPU memory fault" in out
