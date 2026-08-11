"""Tests for compile/utils.py."""

import sys

import pytest

from flashinfer_bench.compile.utils import (
    create_package_name,
    get_build_target_tag,
    write_sources_to_path,
)
from flashinfer_bench.data import BuildSpec, Solution, SourceFile, SupportedLanguages


def test_write_sources_to_path(tmp_path):
    """Test that write_sources_to_path creates files correctly."""
    sources = [
        SourceFile(path="main.py", content="print('hello')"),
        SourceFile(path="pkg/helper.py", content="def helper(): pass"),
    ]

    paths = write_sources_to_path(tmp_path, sources)

    assert len(paths) == 2
    assert (tmp_path / "main.py").exists()
    assert (tmp_path / "main.py").read_text() == "print('hello')"
    assert (tmp_path / "pkg" / "helper.py").exists()
    assert (tmp_path / "pkg" / "helper.py").read_text() == "def helper(): pass"


def test_create_package_name():
    """Test package name creation."""
    solution = Solution(
        name="my_solution",
        definition="test_def",
        author="test",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run(): pass")],
    )

    package_name = create_package_name(solution, "fib_python_")

    # Should start with prefix
    assert package_name.startswith("fib_python_")
    # Should contain normalized solution name
    assert "my_solution" in package_name
    # Should end with hash
    assert len(package_name.split("_")[-1]) == 6  # 6-char hash


def test_create_package_name_normalization():
    """Test that special characters are normalized to underscores."""
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
    )
    sources = [SourceFile(path="main.py", content="def run(): pass")]

    solution = Solution(
        name="my-solution.v1@test", definition="test_def", author="test", spec=spec, sources=sources
    )
    package_name = create_package_name(solution, "")
    assert package_name.startswith("my_solution_v1_test")

    # Special characters should be replaced with underscores
    assert "-" not in package_name
    assert "." not in package_name
    assert "@" not in package_name

    solution2 = Solution(
        name="123solution", definition="test_def", author="test", spec=spec, sources=sources
    )
    package_name2 = create_package_name(solution2, "")
    assert package_name2.startswith("_123solution")


def test_create_package_name_deterministic():
    """Test that the same solution produces the same package name."""
    spec = BuildSpec(
        language=SupportedLanguages.PYTHON, target_hardware=["cpu"], entry_point="main.py::run"
    )
    solution1 = Solution(
        name="my_solution",
        definition="test_def",
        author="test",
        spec=spec,
        sources=[SourceFile(path="main.py", content="def run(): return 1")],
    )

    name1 = create_package_name(solution1, "prefix_")
    name2 = create_package_name(solution1, "prefix_")

    assert name1 == name2

    solution2 = Solution(
        name="my_solution",
        definition="test_def",
        author="test",
        spec=spec,
        sources=[SourceFile(path="main.py", content="def run(): return 2")],
    )
    name3 = create_package_name(solution2, "prefix_")

    assert name1 != name3


def test_get_build_target_tag_prefers_rocm_arch_env(monkeypatch):
    """TVM_FFI_ROCM_ARCH_LIST wins, since that is the arch the build actually honours."""
    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx942")
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

    assert get_build_target_tag() == "hip_gfx942"


def test_get_build_target_tag_distinguishes_architectures(monkeypatch):
    """Different targets must produce different tags, or the cache cannot separate them."""
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx942")
    gfx942 = get_build_target_tag()
    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx950")
    gfx950 = get_build_target_tag()
    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx942 gfx950")
    both = get_build_target_tag()

    assert len({gfx942, gfx950, both}) == 3
    # Path-segment safe: the multi-arch list must not leak a space or separator.
    assert both == "hip_gfx942_gfx950"


def test_get_build_target_tag_is_filesystem_safe(monkeypatch):
    """gcnArchName carries feature suffixes like ':xnack-' that cannot go into a path raw."""
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx942:sramecc+:xnack-")

    tag = get_build_target_tag()

    assert tag == "hip_gfx942_sramecc_plus_xnack_minus"
    assert set(tag) <= set("abcdefghijklmnopqrstuvwxyz0123456789_")


def test_get_build_target_tag_preserves_feature_polarity(monkeypatch):
    """xnack+ and xnack- are distinct code objects, so they must not share a cache key.

    Regression guard: sanitizing the tag by deleting punctuation collapsed both suffixes to a
    bare 'xnack', which is exactly the wrong-binary-from-cache bug the tag exists to prevent.
    """
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)

    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx942:xnack+")
    xnack_on = get_build_target_tag()
    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx942:xnack-")
    xnack_off = get_build_target_tag()

    assert xnack_on != xnack_off


def test_native_builders_segregate_cache_by_target(monkeypatch, tmp_path):
    """A gfx942 .so must not be served from cache to a gfx950 run.

    Regression guard: the cache key is Solution.hash(), which is identical across architectures,
    so without a target segment in the path the second call returns the first build's artifact
    and the kernel fails at launch with hipErrorNoBinaryForGpu.
    """
    from flashinfer_bench.compile.builders.tvm_ffi_builder import TVMFFIBuilder

    monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path))
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    solution = Solution(
        name="sol",
        definition="def",
        author="ut",
        spec=BuildSpec(
            language=SupportedLanguages.CUDA, target_hardware=["rocm"], entry_point="k.cu::run"
        ),
        sources=[SourceFile(path="k.cu", content="// kernel")],
    )
    builder = TVMFFIBuilder()

    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx942")
    name_942, path_942 = builder._get_package_name_and_build_path(solution)
    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx950")
    name_950, path_950 = builder._get_package_name_and_build_path(solution)

    # Same solution, so the same package name -- only the target segment may differ.
    assert name_942 == name_950
    assert path_942 != path_950
    assert "gfx942" in str(path_942) and "gfx950" in str(path_950)


def test_python_builder_cache_stays_target_independent(monkeypatch, tmp_path):
    """PythonBuilder only stages source, so its cache must stay portable across machines."""
    from flashinfer_bench.compile.builders.python_builder import PythonBuilder

    monkeypatch.setenv("FIB_CACHE_PATH", str(tmp_path))
    solution = Solution(
        name="sol",
        definition="def",
        author="ut",
        spec=BuildSpec(
            language=SupportedLanguages.PYTHON, target_hardware=["rocm"], entry_point="main.py::run"
        ),
        sources=[SourceFile(path="main.py", content="def run():\n    return 1\n")],
    )
    builder = PythonBuilder()

    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx942")
    _, path_942 = builder._get_package_name_and_build_path(solution)
    monkeypatch.setenv("TVM_FFI_ROCM_ARCH_LIST", "gfx950")
    _, path_950 = builder._get_package_name_and_build_path(solution)

    assert path_942 == path_950


if __name__ == "__main__":
    pytest.main(sys.argv)
