"""TVM-FFI based builder for CUDA kernels with automatic caching. This is the primary builder for
CUDA and C++ kernels."""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, ClassVar, List, Optional, Tuple

from flashinfer_bench.compile.builder import Builder, BuildError
from flashinfer_bench.compile.runnable import Runnable, RunnableMetadata
from flashinfer_bench.compile.utils import write_sources_to_path
from flashinfer_bench.data import Definition, Solution, SupportedBindings, SupportedLanguages

logger = logging.getLogger(__name__)

# File extension mappings for source file classification
_CUDA_EXTENSIONS: List[str] = [".cu"]  # CUDA source files
_CPP_EXTENSIONS: List[str] = [".cpp", ".cc", ".cxx", ".c"]  # C/C++ source files


class TVMFFIBuilder(Builder):
    """Builder using TVM-FFI with automatic caching and supports multi-process and multi-threaded
    compilation. The result is framework agnostic and supports DLPack interop with PyTorch, JAX,
    etc.

    Cache logic: If the builder is asked to build the same solution again, it will return the cached
    result. If another builder is asking to build the same solution, as long as the build directory
    exists, it will return the cached result.

    The solution to compile should be written in destination-passing style, i.e. the function
    should take the input tensors and the output tensors as arguments.

    Examples
    --------
    >>> builder = TVMFFIBuilder()
    >>> runnable = builder.build(definition, solution)
    >>> output = runnable(x=input_tensor)  # Allocates and returns output
    >>> runnable.call_dest(x=input_tensor, output=output_tensor)  # Destination-passing style
    """

    _PACKAGE_PREFIX: ClassVar[str] = "tvm_ffi_"
    """Prefix for cache keys to avoid collisions with other builders"""

    _BUILD_DIR_NAME: ClassVar[str] = "tvm_ffi"
    """Subdirectory under FIB_CACHE_PATH where build artifacts are stored"""

    _LOCK_FILE_NAME: ClassVar[str] = "flashinfer_bench_tvm_ffi_lock"
    """File lock name for multi-process synchronization during compilation"""

    def __init__(self) -> None:
        """Initialize the TVMFFIBuilder."""
        super().__init__(self._PACKAGE_PREFIX, self._BUILD_DIR_NAME)

    def _is_target_specific(self) -> bool:
        """True: hipcc/nvcc compile this solution for one ``--offload-arch``/``-arch`` target."""
        return True

    # BLAS dependency names a solution may declare (the CUDA name `cublas` is accepted and mapped).
    # Only hipBLAS/rocBLAS are linked below; hipBLASLt (a separate lib/header) is not wired yet.
    _BLAS_DEPS: ClassVar[List[str]] = ["cublas", "hipblas", "rocblas"]

    @staticmethod
    def _is_hip_backend() -> bool:
        """Whether tvm-ffi will use the HIP (ROCm) backend, mirroring its own auto-detection.

        HIP if torch is a ROCm build (``torch.version.hip``), or if ``hipcc`` is available and
        ``nvcc`` is not. Used to gate the ROCm-specific hipify/BLAS handling so it never runs when a
        CUDA (nvcc) toolchain would actually be selected.
        """
        try:
            import torch

            if getattr(torch.version, "hip", None):
                return True
        except Exception:
            pass
        return shutil.which("hipcc") is not None and shutil.which("nvcc") is None

    @staticmethod
    def _find_rocm_lib_path() -> Optional[str]:
        """Find the ROCm library directory (hipblas, rocblas, etc.).

        Honors ROCM_PATH / HIP_PATH, then falls back to the conventional /opt/rocm. Checks both
        ``lib`` and ``lib64`` and accepts a versioned ``libamdhip64.so*`` soname.
        """
        roots: List[Path] = []
        for env in ("ROCM_PATH", "HIP_PATH"):
            root = os.environ.get(env)
            if root:
                roots.append(Path(root))
        roots.append(Path("/opt/rocm"))
        for root in roots:
            for sub in ("lib", "lib64"):
                lib_dir = root / sub
                if any(lib_dir.glob("libamdhip64.so*")):
                    return str(lib_dir)
        return None

    def _needs_blas(self, solution: Solution) -> bool:
        """True if the solution declares a BLAS dependency (cublas/hipblas/rocblas)."""
        deps = [d.lower() for d in (solution.spec.dependencies or [])]
        return any(d in self._BLAS_DEPS for d in deps)

    @staticmethod
    def is_available() -> bool:
        """Check if TVM-FFI is available in the current environment."""
        try:
            import tvm_ffi  # noqa: F401
        except ImportError:
            return False
        return True

    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can build the given solution. The solution should be CUDA or
        C++ source code with TVM-FFI binding (or no binding specified, which defaults to TVM-FFI).

        Parameters
        ----------
        solution : Solution
            Solution to check

        Returns
        -------
        bool
            True if solution language is CUDA or C++ and binding is TVM-FFI or None
        """
        is_cpp_or_cuda = (
            solution.spec.language == SupportedLanguages.CUDA
            or solution.spec.language == SupportedLanguages.CPP
        )
        is_tvm_ffi_binding = (
            solution.spec.binding is None or solution.spec.binding == SupportedBindings.TVM_FFI
        )
        return is_cpp_or_cuda and is_tvm_ffi_binding

    def _check_sources(self, path: Path, key: str, solution: Solution) -> bool:
        """Check if the source code is vaild, and if the cached .so can be used by comparing source
        files and .so existence.

        Returns True (can use cached .so) only if:
        1. The compiled .so file exists
        2. All source files exist with identical content

        Parameters
        ----------
        path : Path
            Build directory path
        key : str
            Unique key for this solution (used to find .so file)
        solution : Solution
            Solution containing source files

        Returns
        -------
        can_use_cached : bool
            True if the cached .so can be used, False if compilation is needed
        """
        # Check if build directory exists
        if not path.exists():
            return False
        elif not path.is_dir():
            raise BuildError(f"Build directory exists but is not a directory: {path}")

        # Check if .so exists
        so_path = path / f"{key}.so"
        if not so_path.is_file():
            return False

        # Check if all files exist and content is identical
        for src in solution.sources:
            # Defensive assertion: the path in the solution should be validated by the Solution
            # model validator, but we add this defensive assertion to be safe.
            src_path_obj = Path(src.path)
            assert not src_path_obj.is_absolute(), f"Absolute path detected: {src.path}"
            assert ".." not in src_path_obj.parts, f"Path traversal detected: {src.path}"

            src_path = path / src.path

            if not src_path.exists():
                return False
            elif not src_path.is_file():
                raise BuildError(f"Source path exists but is not a file: {src_path}")

            if src_path.read_text() != src.content:
                return False

        # All checks passed: can use cached .so
        return True

    def _filter_sources(self, source_paths: List[Path]) -> Tuple[List[str], List[str]]:
        """Filter source files by extension into C++ and CUDA source file paths.

        Parameters
        ----------
        source_paths : List[Path]
            List of source file paths.

        Returns
        -------
        cpp_files : List[str]
            List of C++ source file paths
        cuda_files : List[str]
            List of CUDA source file paths
        """
        cpp_files: List[str] = []
        cuda_files: List[str] = []
        for src_path in source_paths:
            if src_path.suffix in _CPP_EXTENSIONS:
                cpp_files.append(str(src_path))
            elif src_path.suffix in _CUDA_EXTENSIONS:
                cuda_files.append(str(src_path))

        return cpp_files, cuda_files

    def _hipify_sources(self, source_paths: List[Path], build_path: Path) -> List[Path]:
        """Translate CUDA sources to HIP so CUDA-authored solutions compile on ROCm.

        Unlike PyTorch's cpp_extension, tvm-ffi compiles ``.cu`` directly with ``hipcc -x hip`` and
        does NOT run hipify, so CUDA-API source (e.g. ``#include <cuda_runtime.h>``, ``cudaMalloc``)
        fails. This translates every source with ``hipify-perl`` into a parallel ``_hipified/``
        directory (preserving relative paths so cross-file includes resolve) and returns the new
        paths. Hipify is idempotent on HIP-native source. The original written sources are left
        untouched so the .so caching check (which compares them to ``solution.content``) still works.

        If ``hipify-perl`` is unavailable, returns the original paths unchanged (best-effort).
        """
        hipify = shutil.which("hipify-perl")
        if hipify is None:
            logger.warning("hipify-perl not found; compiling sources without CUDA->HIP translation")
            return source_paths

        hip_root = build_path / "_hipified"
        out_paths: List[Path] = []
        for src in source_paths:
            rel = src.relative_to(build_path)
            dst = hip_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = subprocess.run(
                    [hipify, str(src)], capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    dst.write_text(result.stdout)
                else:
                    # Log the real hipify error so a later hipcc failure is diagnosable, then fall
                    # back to the original source (best-effort).
                    logger.warning(
                        "hipify-perl exited %d for %s; using original source. stderr:\n%s",
                        result.returncode,
                        src,
                        result.stderr,
                    )
                    dst.write_text(src.read_text())
            except Exception:
                logger.warning(
                    "hipify-perl failed for %s; using original source", src, exc_info=True
                )
                dst.write_text(src.read_text())
            out_paths.append(dst)
        return out_paths

    def _get_entry_symbol(self, solution: Solution) -> str:
        """Extract function symbol from entry_point.

        Parameters
        ----------
        solution : Solution
            Solution with entry_point in format 'file.ext::symbol'

        Returns
        -------
        str
            The function symbol name to be loaded from the compiled module

        Raises
        ------
        BuildError
            If entry_point format is invalid (missing '::' separator)
        """
        entry_point = solution.spec.entry_point
        if "::" not in entry_point:
            raise BuildError(
                f"Invalid entry_point format: {entry_point}. Expected 'file.extension::symbol'"
            )
        return entry_point.split("::")[-1]

    def _get_cleaner(self, build_path: Path) -> Callable[[], None]:
        """Get a cleaner function for the build directory. It will remove the build directory.

        Parameters
        ----------
        build_path : Path
            The path to the build directory

        Returns
        -------
        callable
            A function that cleans up the build directory.
        """

        def cleaner() -> None:
            shutil.rmtree(build_path, ignore_errors=True)

        return cleaner

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build with automatic caching - compile once, load from cache afterwards.

        This method implements intelligent caching:
        1. Checks if a compiled .so file already exists
        2. If not, writes source files and compiles them
        3. Loads the compiled module (from cache or fresh build)
        4. Returns a runnable wrapper

        The caching is multi-process safe, enabling efficient parallel benchmarking.

        Parameters
        ----------
        definition : Definition
            Problem definition specifying inputs/outputs
        solution : Solution
            Solution containing source code and build specification

        Returns
        -------
        Runnable
            A runnable wrapper around the compiled TVM-FFI module that supports both
            value-returning style (via __call__) and destination-passing style (via call_dps)

        Raises
        ------
        BuildError
            If compilation fails, module loading fails, or entry point is invalid
        """
        import tvm_ffi
        from tvm_ffi.utils import FileLock

        package_name, build_path = self._get_package_name_and_build_path(solution)
        entry_symbol = self._get_entry_symbol(solution)
        can_use_cached = self._check_sources(build_path, package_name, solution)

        # Check if cached .so can be used. If not, build the solution.
        # This check and build are thread-safe through the FileLock
        if can_use_cached:
            output_lib_path = str(build_path / f"{package_name}.so")
        else:
            # Ensure build directory exists before creating file lock
            build_path.mkdir(parents=True, exist_ok=True)
            with FileLock(build_path / self._LOCK_FILE_NAME):
                # Double-check after acquiring lock (another process may have built it)
                if self._check_sources(build_path, package_name, solution):
                    output_lib_path = str(build_path / f"{package_name}.so")
                else:
                    src_paths = write_sources_to_path(build_path, solution.sources)
                    extra_include_paths = [str(build_path)]
                    extra_ldflags: List[str] = []
                    # ROCm-specific handling is gated on tvm-ffi actually selecting the HIP backend,
                    # so it never runs under a CUDA (nvcc) toolchain (where hipified sources would
                    # fail to compile).
                    if self._is_hip_backend():
                        # Translate CUDA -> HIP so CUDA-authored solutions compile under tvm-ffi's
                        # hipcc backend (idempotent on HIP-native source). Originals are kept for the
                        # .so cache check; compilation uses the hipified copies.
                        hipified_paths = self._hipify_sources(src_paths, build_path)
                        cpp_files, cuda_files = self._filter_sources(hipified_paths)
                        extra_include_paths.append(str(build_path / "_hipified"))
                        # tvm-ffi links the HIP runtime (libamdhip64) itself, so we add no GPU-runtime
                        # flags. When a solution declares a BLAS dependency we add hipBLAS/rocBLAS
                        # (the ROCm equivalents of cuBLAS). hipify rewrites <cublas_v2.h> to
                        # <hipblas.h>, but the header lives in include/hipblas/, so those subdirs are
                        # added to the include path.
                        if self._needs_blas(solution):
                            rocm_lib_path = self._find_rocm_lib_path()
                            if rocm_lib_path:
                                extra_ldflags.append(f"-L{rocm_lib_path}")
                                rocm_root = Path(rocm_lib_path).parent
                                for sub in ("include/hipblas", "include/rocblas"):
                                    inc = rocm_root / sub
                                    if inc.is_dir():
                                        extra_include_paths.append(str(inc))
                            extra_ldflags += ["-lhipblas", "-lrocblas"]
                    else:
                        # Non-HIP backend (unmaintained CUDA path in this ROCm fork): compile the
                        # original sources without hipify.
                        cpp_files, cuda_files = self._filter_sources(src_paths)
                    try:
                        # Compile sources to shared library
                        output_lib_path = tvm_ffi.cpp.build(
                            name=package_name,
                            cpp_files=cpp_files,
                            cuda_files=cuda_files,
                            extra_include_paths=extra_include_paths,
                            extra_ldflags=extra_ldflags,
                            build_directory=build_path,
                        )
                    except Exception as e:
                        raise BuildError(
                            f"TVM-FFI compilation failed for '{solution.name}': {e}"
                        ) from e

        # Validate all symbols resolve before loading via tvm_ffi
        try:
            _probe = ctypes.CDLL(output_lib_path, mode=os.RTLD_NOW)
            del _probe
        except OSError as e:
            raise BuildError(f"Shared library has unresolved symbols: {e}") from e

        # Load the compiled module
        try:
            mod = tvm_ffi.load_module(output_lib_path)
        except Exception as e:
            raise BuildError(f"Failed to load compiled module: {e}") from e

        # Create metadata for the runnable
        metadata = RunnableMetadata(
            build_type="tvm_ffi",
            definition_name=definition.name,
            solution_name=solution.name,
            destination_passing_style=solution.spec.destination_passing_style,
            definition=definition,
            misc={"entry_symbol": entry_symbol, "binary": output_lib_path},
        )

        try:
            callable = getattr(mod, entry_symbol)
        except AttributeError as e:
            raise BuildError(f"Entry point '{entry_symbol}' not found in module") from e

        self._try_validate_signature(callable, definition, solution)

        cleaner = self._get_cleaner(build_path)
        return Runnable(callable=callable, metadata=metadata, cleaner=cleaner)
