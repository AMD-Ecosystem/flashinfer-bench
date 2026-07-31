import importlib.util
from pathlib import Path
from typing import List

import pytest


def _torch_cuda_available() -> bool:
    """Check if CUDA is available from PyTorch.

    Returns
    -------
    bool
        True if CUDA is available from PyTorch, False otherwise.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _flashinfer_available() -> bool:
    """Check if the ``flashinfer`` package is importable.

    On ROCm this is ``amd-flashinfer``, which imports under the same name. It is an optional
    extra rather than a hard dependency, so it is absent on CPU-only environments such as CI.

    Returns
    -------
    bool
        True if ``flashinfer`` can be imported, False otherwise.
    """
    try:
        return importlib.util.find_spec("flashinfer") is not None
    except (ImportError, ValueError):
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Skip tests whose environment prerequisites (CUDA, flashinfer) are unavailable."""
    skip_cuda = (
        None
        if _torch_cuda_available()
        else pytest.mark.skip(reason="CUDA not available from PyTorch, skip test")
    )
    skip_flashinfer = (
        None
        if _flashinfer_available()
        else pytest.mark.skip(reason="flashinfer (amd-flashinfer on ROCm) not installed")
    )
    if skip_cuda is None and skip_flashinfer is None:
        return

    for item in items:
        if skip_cuda is not None and any(item.iter_markers(name="requires_torch_cuda")):
            item.add_marker(skip_cuda)
        if skip_flashinfer is not None and any(item.iter_markers(name="requires_flashinfer")):
            item.add_marker(skip_flashinfer)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use isolated temporary directory for cache in all tests.

    This fixture automatically sets FIB_CACHE_PATH to a unique temporary
    directory for each test, preventing cache pollution between tests.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FIB_CACHE_PATH", str(cache_dir))
    return cache_dir
