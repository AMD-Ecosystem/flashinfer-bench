"""Tests for trace-file discovery in `scripts/collect_stream.py`.

``test_finds_trace_under_author_directory`` fails against the pre-fix code, which reconstructed
``traces/{op_type}/{def_name}.jsonl`` — the pre-#379 layout, missing the author segment. It matched
nothing, so the upload step logged a warning, returned, and let the pipeline report success having
uploaded no traces.

`scripts/` is not an installed package, so the module is loaded by path. Its module-level imports
are all stdlib (``huggingface_hub`` is imported lazily inside ``push_trace``), so this is cheap.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "collect_stream.py"


def _load_collect_stream():
    spec = importlib.util.spec_from_file_location("collect_stream_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collect_stream = _load_collect_stream()


def _write_trace(trace_dir: Path, *parts: str) -> Path:
    """Create a trace JSONL at trace_dir/traces/<parts...> and return it."""
    path = trace_dir.joinpath("traces", *parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"definition": "d1"}\n')
    return path


def test_finds_trace_under_author_directory(tmp_path: Path):
    """The real layout is traces/{author}/{op_type}/{def_name}.jsonl."""
    expected = _write_trace(tmp_path, "baseline", "rmsnorm", "d1.jsonl")

    found = collect_stream.find_trace_file(tmp_path, "d1")

    assert found == expected
    # The dataset path must keep the author segment, not flatten it away.
    assert found.relative_to(tmp_path).as_posix() == "traces/baseline/rmsnorm/d1.jsonl"


def test_missing_trace_raises_rather_than_silently_skipping(tmp_path: Path):
    (tmp_path / "traces").mkdir()

    with pytest.raises(FileNotFoundError, match="d1.jsonl"):
        collect_stream.find_trace_file(tmp_path, "d1")


def test_ambiguous_trace_raises(tmp_path: Path):
    """Two authors produced traces for the same definition — refuse to guess."""
    _write_trace(tmp_path, "alice", "rmsnorm", "d1.jsonl")
    _write_trace(tmp_path, "bob", "rmsnorm", "d1.jsonl")

    with pytest.raises(ValueError, match="multiple trace files"):
        collect_stream.find_trace_file(tmp_path, "d1")


def test_does_not_match_a_different_definition(tmp_path: Path):
    _write_trace(tmp_path, "baseline", "rmsnorm", "d2.jsonl")

    with pytest.raises(FileNotFoundError):
        collect_stream.find_trace_file(tmp_path, "d1")
