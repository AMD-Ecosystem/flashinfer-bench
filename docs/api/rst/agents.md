# flashinfer_bench.agents

`flashinfer_bench.agents` provides tools for kernel agent development and debugging.
This module provides the following tools:

1. **Profiling Tools**: Run `rocprofv3` and a best-effort memory checker on solutions
2. **FFI Prompts**: Provide context about the FlashInfer Bench API for LLM agents

On this ROCm fork the agent-facing profiler is `rocprofv3`; {py:func}`~flashinfer_bench.agents.get_all_tool_schemas`
exposes the rocprof pair and not the NCU one. The NCU entry points remain importable for
upstream parity but are NVIDIA-only and will report a missing executable on AMD.

This package also provides JSON Schema version of the tools by calling {py:func}`~flashinfer_bench.agents.function_to_schema` and {py:func}`~flashinfer_bench.agents.get_all_tool_schemas`.

```{eval-rst}
.. currentmodule:: flashinfer_bench.agents

.. autofunction:: flashinfer_bench_run_rocprof

.. autofunction:: flashinfer_bench_list_rocprof_options

.. autofunction:: flashinfer_bench_run_sanitizer

.. autofunction:: flashinfer_bench_run_ncu

.. autofunction:: flashinfer_bench_list_ncu_options

.. autofunction:: extract_solution_to_files

.. autofunction:: pack_solution_from_files

.. autofunction:: function_to_schema

.. autofunction:: get_all_tool_schemas

.. automodule:: flashinfer_bench.agents.ffi_prompt
   :members:
   :no-value:
```
