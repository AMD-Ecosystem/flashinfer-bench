"""Tests for `flashinfer-bench report merge` (`cli.main.merge_trace_sets` / `export_trace_set`).

These fail against the pre-fix code in three distinct ways: ``AttributeError: 'TraceSet' object
has no attribute 'workload'`` (the field is ``workloads``); the ``assert not
t.is_workload_trace()`` in ``TraceSet.from_path``, because merge wrote workload traces into
``traces/`` where the loader expects only execution traces; and ``get_solution`` returning ``None``
for solutions merged in after the first TraceSet, whose lookup indexes were never rebuilt.
"""

from pathlib import Path

import pytest

from flashinfer_bench.cli.main import export_trace_set, merge_trace_sets
from flashinfer_bench.data import (
    AxisVar,
    BuildSpec,
    Correctness,
    Definition,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    RandomInput,
    Solution,
    SourceFile,
    SupportedLanguages,
    TensorSpec,
    Trace,
    TraceSet,
    Workload,
)


def _definition(name: str) -> Definition:
    return Definition(
        name=name,
        op_type="op",
        axes={"M": AxisVar()},
        inputs={"A": TensorSpec(shape=["M"], dtype="float32")},
        outputs={"B": TensorSpec(shape=["M"], dtype="float32")},
        reference="def run(a):\n    return a\n",
    )


def _solution(name: str, def_name: str, author: str) -> Solution:
    return Solution(
        name=name,
        definition=def_name,
        author=author,
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run():\n    pass\n")],
    )


def _workload_trace(def_name: str, uuid: str) -> Trace:
    """A workload-only trace: no solution, no evaluation."""
    return Trace(
        definition=def_name,
        workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid=uuid),
    )


def _execution_trace(def_name: str, solution: str, uuid: str) -> Trace:
    return Trace(
        definition=def_name,
        workload=Workload(axes={"M": 2}, inputs={"A": RandomInput()}, uuid=uuid),
        solution=solution,
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            log="log",
            environment=Environment(hardware="cpu"),
            timestamp="t",
            correctness=Correctness(max_relative_error=0.0, max_absolute_error=0.0),
            performance=Performance(latency_ms=1.0, reference_latency_ms=2.0, speedup_factor=2.0),
        ),
    )


def _trace_set(def_name: str, author: str, uuid_prefix: str) -> TraceSet:
    # Populate via the constructor, not by assigning the dicts afterwards, so __post_init__ builds
    # the lookup indexes — otherwise the inputs start out as stale as the bug under test.
    solution_name = f"s_{author}"
    return TraceSet(
        definitions={def_name: _definition(def_name)},
        solutions={def_name: [_solution(solution_name, def_name, author)]},
        workloads={def_name: [_workload_trace(def_name, f"{uuid_prefix}_w")]},
        traces={def_name: [_execution_trace(def_name, solution_name, f"{uuid_prefix}_t")]},
    )


def test_merge_trace_sets_combines_workloads():
    """Workloads from every input TraceSet reach the merged result.

    Regression: the merge loop read ``trace_set.workload``; the field is ``workloads``. Python
    raises ``AttributeError`` on the first non-first input, so merging any two TraceSets failed.
    """
    first = _trace_set("d1", "a", "u1")
    second = _trace_set("d2", "b", "u2")

    merged = merge_trace_sets([first, second])

    assert set(merged.definitions) == {"d1", "d2"}
    assert set(merged.workloads) == {"d1", "d2"}
    assert [t.workload.uuid for t in merged.workloads["d1"]] == ["u1_w"]
    assert [t.workload.uuid for t in merged.workloads["d2"]] == ["u2_w"]


def test_merge_trace_sets_concatenates_shared_definition():
    """Two TraceSets carrying the same definition have their workloads and traces concatenated."""
    first = _trace_set("d1", "a", "u1")
    second = _trace_set("d1", "b", "u2")
    # Same definition on both sides, so the merge must not raise a conflict.
    second.definitions["d1"] = first.definitions["d1"]

    merged = merge_trace_sets([first, second])

    assert sorted(t.workload.uuid for t in merged.workloads["d1"]) == ["u1_w", "u2_w"]
    assert sorted(t.workload.uuid for t in merged.traces["d1"]) == ["u1_t", "u2_t"]
    assert len(merged.solutions["d1"]) == 2


def test_merged_lookup_indexes_cover_every_input():
    """``get_solution`` must find solutions contributed by TraceSets after the first.

    The merge loops mutate ``solutions``/``traces`` directly, so the indexes built at construction
    only ever held the first TraceSet's entries. ``merged.get_solution("s_b")`` returned ``None``
    for a solution sitting in ``merged.solutions`` — the merged object was only usable after a
    save + reload round-trip.
    """
    merged = merge_trace_sets([_trace_set("d1", "a", "u1"), _trace_set("d2", "b", "u2")])

    assert merged.get_solution("s_a") is not None
    assert merged.get_solution("s_b") is not None
    # The trace index backs the score/ranking helpers and must cover later inputs too.
    assert merged._traces_by_solution.keys() == {"s_a", "s_b"}


def test_merge_dedupes_an_identical_shared_solution():
    """Inputs commonly share a solution (a baseline); that must merge, not explode."""
    first = _trace_set("d1", "a", "u1")
    second = _trace_set("d1", "a", "u2")
    second.definitions["d1"] = first.definitions["d1"]

    merged = merge_trace_sets([first, second])

    assert [s.name for s in merged.solutions["d1"]] == ["s_a"]
    assert merged.get_solution("s_a") is not None
    # Both sides' traces still land.
    assert sorted(t.workload.uuid for t in merged.traces["d1"]) == ["u1_t", "u2_t"]


def test_merge_raises_on_conflicting_solution_of_the_same_name():
    """Same name, different implementation: refuse rather than emit a set from_path rejects.

    ``Solution`` equality is a content hash over definition/spec/sources that deliberately
    excludes name, author and description — so the conflict has to be a real source difference.
    """
    first = _trace_set("d1", "a", "u1")
    second = _trace_set("d1", "a", "u2")
    second.definitions["d1"] = first.definitions["d1"]
    conflicting = Solution(
        name="s_a",
        definition="d1",
        author="a",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run():\n    return 1\n")],
    )
    assert conflicting != first.solutions["d1"][0]
    second.solutions["d1"] = [conflicting]

    with pytest.raises(ValueError, match="Solution conflict for 's_a'"):
        merge_trace_sets([first, second])


def test_merge_output_round_trips_through_from_path(tmp_path: Path):
    """The exported directory must load back via ``TraceSet.from_path``.

    Regression: workload traces were written to ``traces/{def}_workloads.jsonl``, but
    ``from_path`` loads workloads from ``workloads/`` and asserts everything under ``traces/`` is
    an execution trace. Merge output was therefore unloadable.
    """
    merged = merge_trace_sets([_trace_set("d1", "a", "u1"), _trace_set("d2", "b", "u2")])

    out = tmp_path / "merged"
    export_trace_set(merged, out)

    reloaded = TraceSet.from_path(str(out))

    assert set(reloaded.definitions) == {"d1", "d2"}
    assert sorted(t.workload.uuid for ts in reloaded.workloads.values() for t in ts) == [
        "u1_w",
        "u2_w",
    ]
    assert sorted(t.workload.uuid for ts in reloaded.traces.values() for t in ts) == [
        "u1_t",
        "u2_t",
    ]
    assert reloaded.get_solution("s_a") is not None
    assert reloaded.get_solution("s_b") is not None


def test_exported_workloads_use_the_op_type_layout(tmp_path: Path):
    """Workloads must export as workloads/{op_type}/{def}.jsonl.

    ``from_path`` rglobs and so would accept a flat file, but ``add_workload_traces`` writes the
    op_type segment and consumers build that path directly — a flat export is invisible to them.
    """
    merged = merge_trace_sets([_trace_set("d1", "a", "u1")])

    out = tmp_path / "merged"
    export_trace_set(merged, out)

    # _definition() uses op_type "op".
    assert (out / "workloads" / "op" / "d1.jsonl").is_file()
    assert not (out / "workloads" / "d1.jsonl").exists()
