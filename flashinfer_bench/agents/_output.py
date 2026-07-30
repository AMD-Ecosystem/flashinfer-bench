"""Shared output helpers for the agent tools.

Each agent tool caps the text it hands back so a large profiler dump or runner stderr cannot
swamp an LLM's context. Keeping that in one place avoids the three tools drifting apart on the
edge cases (``max_lines=0``, marker wording), which is what happened when they each had a copy.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["truncate"]


def truncate(text: str, max_lines: Optional[int]) -> str:
    """Cap ``text`` at ``max_lines`` lines, noting how many were dropped.

    ``max_lines=None`` means no limit. ``max_lines=0`` keeps nothing and returns the marker on its
    own — joining an empty head would otherwise open the output with a stray blank line, which
    shows up as ``"ERROR:\\n\\n[...]"`` on the tools' error paths.
    """
    if max_lines is None:
        return text
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    head = "\n".join(lines[:max_lines])
    marker = f"[... {len(lines) - max_lines} more lines]"
    return f"{head}\n{marker}" if head else marker
