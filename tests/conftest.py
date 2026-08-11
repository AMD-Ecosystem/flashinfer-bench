import importlib.util
from pathlib import Path
from typing import List, Sequence

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
    """Check whether the ``flashinfer`` package is installed.

    On ROCm this is ``amd-flashinfer``, which imports under the same name. It is an optional
    extra rather than a hard dependency, so it is absent on CPU-only environments such as CI.

    This probes for the module spec rather than importing: importing flashinfer is expensive and
    can touch the GPU, which is not something to do during collection. So this answers "is it
    installed", not "does importing it succeed" — an installed-but-broken flashinfer reports True
    and its tests fail rather than skip. That is deliberate: a broken install should be visible,
    not silently skipped.

    Returns
    -------
    bool
        True if ``flashinfer`` is installed, False otherwise.
    """
    try:
        return importlib.util.find_spec("flashinfer") is not None
    except (ImportError, ValueError):
        return False


def _missing_flashinfer_apis(names: Sequence[str]) -> List[str]:
    """Return the subset of ``names`` that the installed flashinfer does not expose.

    ``flashinfer`` being installed does not mean it implements the whole upstream API: the ROCm
    build (``amd-flashinfer``) ships a subset, so e.g. ``flashinfer.mla`` is absent there while
    the NVIDIA package has it. Tests that reach for such an API need a skip keyed on the attribute
    rather than on the package, otherwise they fail on every AMD box.

    This uses ``hasattr`` on the imported package on purpose — that is exactly what the tests do
    (``import flashinfer`` then ``flashinfer.mla...``), so it stays true to whether the test can
    actually run, including for submodules that ``__init__`` chooses not to re-export.

    An import failure returns no missing names, so a broken install fails its tests instead of
    silently skipping them — the same tradeoff ``_flashinfer_available`` documents.
    """
    try:
        import flashinfer
    except Exception:
        return []
    return [name for name in names if not hasattr(flashinfer, name)]


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Skip tests whose environment prerequisites (CUDA, flashinfer, flashinfer APIs) are absent."""
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
    for item in items:
        if skip_cuda is not None and any(item.iter_markers(name="requires_torch_cuda")):
            item.add_marker(skip_cuda)
        api_markers = list(item.iter_markers(name="requires_flashinfer_api"))
        # requires_flashinfer_api implies requires_flashinfer: asking for an attribute of the
        # package is asking for the package, so the marker stands on its own and a test need not
        # carry both.
        needs_flashinfer = api_markers or any(item.iter_markers(name="requires_flashinfer"))
        if skip_flashinfer is not None and needs_flashinfer:
            item.add_marker(skip_flashinfer)
            # The package is absent, so probing it for individual APIs would only import-fail.
            continue
        for marker in api_markers:
            missing = _missing_flashinfer_apis(marker.args)
            if missing:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            "installed flashinfer does not provide: "
                            f"{', '.join(f'flashinfer.{name}' for name in missing)}"
                        )
                    )
                )


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
