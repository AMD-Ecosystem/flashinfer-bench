"""Tests for `flashinfer-bench report merge` (`cli.main.merge_trace_sets` / `export_trace_set`).

Both tests fail against the pre-fix code: the first with ``AttributeError: 'TraceSet' object has
no attribute 'workload'``, the second with the ``assert not t.is_workload_trace()`` in
``TraceSet.from_path`` — merge wrote workload traces into ``traces/``, where the loader expects
only execution traces.
"""

from pathlib import Path

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
    ts = TraceSet()
    ts.definitions[def_name] = _definition(def_name)
    ts.solutions[def_name] = [_solution(f"s_{author}", def_name, author)]
    ts.workloads[def_name] = [_workload_trace(def_name, f"{uuid_prefix}_w")]
    ts.traces[def_name] = [_execution_trace(def_name, f"s_{author}", f"{uuid_prefix}_t")]
    return ts


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
