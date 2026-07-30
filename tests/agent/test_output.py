"""Tests for the shared agent-tool output helpers.

These used to live in test_rocprof.py, when each tool carried its own copy of the truncation
logic. The copies drifted (ncu's emitted a different marker and an unconditional blank line),
so the helper — and its tests — now live in one place.
"""

from __future__ import annotations

from flashinfer_bench.agents._output import truncate


def test_truncate_respects_limit_and_none():
    text = "\n".join(f"line{i}" for i in range(10))

    assert truncate(text, None) == text  # None means no limit
    assert truncate(text, 20) == text  # under the limit, unchanged

    capped = truncate(text, 3)
    assert capped.split("\n")[:3] == ["line0", "line1", "line2"]
    assert "7 more lines" in capped


def test_truncate_at_zero_has_no_leading_blank_line():
    """max_lines=0 keeps an "ERROR:" prefix first AND must not open with a blank line."""
    detail = truncate("boom\nmore", 0)

    assert detail == "[... 2 more lines]"  # marker alone, no leading newline
    rendered = f"ERROR: rocprofv3 exited with code 1:\n{detail}"
    assert rendered.startswith("ERROR:")
    assert "\n\n" not in rendered


def test_profile_region_is_shared_and_tool_agnostic():
    """The emitter and both readers must resolve to one constant.

    If they ever drift, region correlation finds nothing and the report silently widens to every
    kernel on the device instead of failing — so pin it here rather than trusting a grep.
    """
    from flashinfer_bench.agents import _solution_runner, ncu, rocprof
    from flashinfer_bench.agents._profiling import PROFILE_REGION

    assert rocprof.PROFILE_REGION is PROFILE_REGION
    assert ncu.PROFILE_REGION is PROFILE_REGION
    assert _solution_runner.PROFILE_REGION is PROFILE_REGION
    # Named for the job, not for one vendor's profiler — it is read by rocprofv3 and NCU alike.
    assert "ncu" not in PROFILE_REGION


def test_all_agent_tools_share_one_truncate():
    """Guard against the drift that prompted the consolidation: one implementation, not three."""
    from flashinfer_bench.agents import ncu, rocprof, sanitizer

    assert rocprof.truncate is truncate
    assert sanitizer.truncate is truncate
    assert ncu.truncate is truncate
