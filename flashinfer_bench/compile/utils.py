"""Utility functions for building solutions."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from flashinfer_bench.data import Solution, SourceFile


def get_build_target_tag() -> str:
    """Return a short tag identifying the GPU target native code is compiled for.

    Build caches are keyed on ``Solution.hash()``, which covers source and build spec but says
    nothing about the machine doing the build. A ``.so`` compiled with ``--offload-arch=gfx942``
    is not loadable on gfx950, yet it is byte-identical from the cache's point of view — so
    without this tag a shared or copied ``FIB_CACHE_PATH`` hands gfx950 a gfx942 binary. That
    fails at first kernel launch with ``hipErrorNoBinaryForGpu`` and, because the cached artifact
    still looks valid, re-running never rebuilds it.

    The tag prefers the explicit arch-list environment variables the build actually honours,
    falling back to the live device's ``gcnArchName`` (which includes feature suffixes such as
    ``:xnack-``; code objects are built per xnack variant, so they belong in the key).

    Returns
    -------
    str
        A filesystem-safe tag such as ``hip_gfx942`` or ``cuda_sm90``, or ``unknown`` when no
        target can be determined.
    """
    for env_name in ("TVM_FFI_ROCM_ARCH_LIST", "PYTORCH_ROCM_ARCH"):
        value = os.environ.get(env_name)
        if value:
            return _make_target_tag("hip", value)
    value = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if value:
        return _make_target_tag("cuda", value)

    detected = _detect_device_target()
    return _make_target_tag(*detected) if detected else "unknown"


def _detect_device_target() -> Optional[Tuple[str, str]]:
    """Return a ``(backend, target)`` pair for the live device, or None."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return None
    gcn_arch = getattr(props, "gcnArchName", None)
    if gcn_arch:
        return ("hip", gcn_arch)
    return ("cuda", f"sm{props.major}{props.minor}")


def _make_target_tag(backend: str, target: str) -> str:
    """Join a backend name and a raw target string into a filesystem-safe path segment.

    ROCm feature suffixes carry a polarity that selects a *different* code object:
    ``gfx942:xnack+`` and ``gfx942:xnack-`` are not interchangeable. So the polarity is spelled
    out in words before punctuation is collapsed to underscores -- deleting it as punctuation
    would map both variants onto one cache key and reintroduce the very stale-``.so`` failure
    this tag exists to prevent. The same applies to CUDA's ``+PTX`` suffix.
    """
    target = target.replace("+", "_plus_").replace("-", "_minus_")
    target = re.sub(r"[^0-9a-zA-Z]+", "_", target).strip("_").lower()
    return f"{backend}_{target}" if target else backend


def write_sources_to_path(path: Path, sources: List[SourceFile]) -> List[Path]:
    """Write source files to a directory and return their paths. Create path if not exists.

    This function writes all source files from a solution to a specified directory,
    creating subdirectories as needed. It performs security checks to prevent path
    traversal attacks and absolute path injection.

    Parameters
    ----------
    path : Path
        The root directory where source files will be written.
    sources : List[SourceFile]
        The list of source files to write. Each file's path must be relative and
        not contain parent directory references ("..").

    Returns
    -------
    List[Path]
        List of absolute paths to the written files.

    Raises
    ------
    AssertionError
        If any source file has an absolute path or contains path traversal.
    """
    path.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for src in sources:
        # Defensive assertion: path should be validated at Solution creation time
        src_path_obj = Path(src.path)

        assert not src_path_obj.is_absolute(), f"Absolute path detected: {src.path}"
        assert ".." not in src_path_obj.parts, f"Path traversal detected: {src.path}"

        src_path = path / src.path

        # Ensure parent directories exist
        src_path.parent.mkdir(parents=True, exist_ok=True)

        # Write source file
        src_path.write_text(src.content)
        paths.append(src_path)

    return paths


def create_package_name(solution: Solution, package_prefix: str = "") -> str:
    """Generate a unique package name for a solution.

    The package name is constructed from three parts:
    1. A prefix (typically identifying the builder)
    2. The normalized solution name (alphanumeric and underscores only)
    3. A 6-character hash of the solution content

    This ensures the package name is both human-readable and uniquely identifies
    the solution's content.

    Parameters
    ----------
    solution : Solution
        The solution to create a package name for.
    prefix : str, optional
        The prefix to prepend to the package name. Default is empty string.

    Returns
    -------
    str
        A unique package name in the format: {prefix}{normalized_name}_{hash}.

    Examples
    --------
    >>> create_package_name(solution, "fib_python_")
    'fib_python_rmsnorm_v1_a3f2b1'
    """
    # Normalize the solution name
    s = re.sub(r"[^0-9a-zA-Z_]", "_", solution.name)
    if not s or s[0].isdigit():
        s = "_" + s

    return package_prefix + s + "_" + solution.hash()[:6]
