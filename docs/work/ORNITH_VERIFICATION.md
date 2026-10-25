# Ornith-1.5 support: verification record

Raw outputs are in `docs/work/ornith-evidence/`. Benchmarks and KL are in [ORNITH_BENCH.md](ORNITH_BENCH.md).

## Package decision (user, 2026-10-25)
Ornith is GGUF architecture `qwen35moe`, the same as Qwen3.6. The user chose to keep it on the shared `src/qwen35`
engine with defaults gated on the Ornith geometry (n_embd 3072, n_ff 1024), instead of a separate `src/ornith` fork.

## Launch path: API vs llama.cpp (same GGUF, same requests, greedy)
| request | Strata via `strata launch --family ornith --model IQ3_XS --model-path ...` | llama-server --jinja --n-cpu-moe 31 |
|---|---|---|
| chat "capital of France" | 'The capital of France is Paris.' | 'The capital of France is Paris.' |
| tool call (get_weather) | finish tool_calls, get_weather {"city":"Paris"} | finish tool_calls, get_weather {"city":"Paris"} |
| Anthropic /v1/messages | ['Bonjour'] | ['Bonjour'] |
Script: `api.py` (OpenAI chat, OpenAI tools, Anthropic messages). llama.cpp output: `ornith-evidence/api_llamacpp.txt`.
Logits: `tools/qwen35/ornith_kl.py`: Strata mean KL 0.0252, llama.cpp floor 0.0381 (0.66x): PASS.

## Qwen/DeepSeek default paths unchanged
- Qwen3.8-Flash-Next IQ2_XS, `strata` from main and from this branch, 48 greedy tokens, default and `--expert-cache 4`:
  the output token ids are identical in all four runs (`ornith-evidence/qwen38_main_vs_branch.txt`). Decode speed is
  the same: 54.52 vs 54.44 tok/s default, 16.06 vs 16.01 with expert-cache 4.
- DeepSeek: `git diff main -- src/dsv41 sycl metal` is empty. No DeepSeek model on this machine, so it was not run.
- The Qwen3.6 path was not run (no Qwen3.6 GGUF here). Ornith-only behaviour is gated on its geometry (re-encode of
  Q4_0 experts, defaults) or on opt-in env vars. The shared changes enlarge the CPU pool buffers and accept IQ3_S downs
  and mixed gate/up types that were refused before.

## Tests
- `pytest tests/terminal`: main 28 failed / branch 28 failed, 1646 passed, and the failure lists are identical
  (`pytest_failures_main.txt`, `pytest_failures_branch_final.txt`). New: `test_launch_ornith.py` (7 tests, pass).
- ctest (CUDA build, non-GPU tests): main fails 4 (expert_cache_segmented_test, ple_parity, ggml_tq_overlay,
  isa_guard_check), the branch fails 3 of those (`ctest_main.txt`, `ctest_branch.txt`).
