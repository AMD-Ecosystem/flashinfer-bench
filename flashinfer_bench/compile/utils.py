"""Utility functions for building solutions."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from flashinfer_bench.data import Solution, SourceFile

# Backend prefix per arch env var. The two toolchains read *different* variables: torch's
# cpp_extension honours PYTORCH_ROCM_ARCH / TORCH_CUDA_ARCH_LIST, while tvm-ffi honours
# TVM_FFI_ROCM_ARCH_LIST / TVM_FFI_CUDA_ARCH_LIST and ignores torch's entirely. Builders therefore
# pass the subset they actually honour -- see :meth:`Builder._target_env_names`.
_ARCH_ENV_BACKENDS: Dict[str, str] = {
    "TVM_FFI_ROCM_ARCH_LIST": "hip",
    "PYTORCH_ROCM_ARCH": "hip",
    "TVM_FFI_CUDA_ARCH_LIST": "cuda",
    "TORCH_CUDA_ARCH_LIST": "cuda",
}

DEFAULT_ARCH_ENV_NAMES: Tuple[str, ...] = (
    "TVM_FFI_ROCM_ARCH_LIST",
    "PYTORCH_ROCM_ARCH",
    "TVM_FFI_CUDA_ARCH_LIST",
    "TORCH_CUDA_ARCH_LIST",
)
"""Fallback precedence for callers that are not tied to one toolchain."""


def get_build_target_tag(env_names: Sequence[str] = DEFAULT_ARCH_ENV_NAMES) -> str:
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

    Parameters
    ----------
    env_names : Sequence[str], optional
        Arch env vars to consult, in precedence order. Callers should pass only the variables
        their own toolchain reads: a Torch build tagged with TVM-FFI's arch is worse than no tag,
        because flipping ``PYTORCH_ROCM_ARCH`` alone would then leave the tag unchanged and serve
        the previous arch's ``.so``.

    Returns
    -------
    str
        A filesystem-safe tag such as ``hip_gfx942`` or ``cuda_sm90``, or ``unknown`` when no
        target can be determined.
    """
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return _make_target_tag(_ARCH_ENV_BACKENDS[env_name], value)

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
