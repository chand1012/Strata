# Ornith validation — 2026-10-21

This is the initial compatibility validation. Later opt-in optimization results are recorded in
[the performance validation](ORNITH-PERFORMANCE.md). The latest [expert-residency experiment](ORNITH-RESIDENCY.md)
compares Ornith with the Qwen control and documents task-specific gains without promoting a general preset.

Tested on an RTX 3090 Ti (24 GiB VRAM), i7-10700 (8 physical cores) and 64 GB installed RAM
(about 61 GiB usable). Built with CUDA 13.0.48 and GNU 13.3.0, for compute capability 8.6, from upstream
`9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4` plus this branch's patch. The pinned ggml revision is
`3cf03257f219afbe7334045ff7c6a06ac68c627d`.

Both quantizations passed full SHA-256 verification, loaded all 899 tensors, and completed generation with
8K context allocated, 32-token prompt chunks, seven pool workers plus the host thread, MTP generation disabled,
and one request at a time. LlamaSwap was unloaded, reported no running models, and the GPU was idle before each
model test. The tests ran sequentially in systemd scopes limited to 56 GiB with no swap.

| Check | Unsloth UD-IQ3_XXS | Bartowski Q3_K_S |
| --- | --- | --- |
| `17 * 23` | `391` | `391` |
| Python `square(x)` | Correct `return x * x` | Correct `return x * x` |
| Three sentences about the blue sky | Coherent explanation | Coherent explanation |
| Key retrieval from 1,160-token prompt | `ORNITH-724` | `ORNITH-724` |
| Sampled peak GPU memory | 23,466 MiB | 23,462 MiB |
| Pinned expert arena reported by engine | 41.30 GiB | 48.16 GiB |
| Engine process exit | 0 | 0 |

Both memory scopes reached their 56 GiB limit, including reclaimable file cache, and finished without an OOM
or swap. This is not a claim that the models require exactly 56 GiB resident RAM. The GPU cache duplicates some
experts held in the host arena.

The engine reported about 20.6 / 21.1 tokens/s for the short prose responses, and 29.4 / 33.8 tokens/s while
reading the 1,160-token prompt, respectively. These are smoke-test timings, not controlled performance
benchmarks. Startup and caching were affected by preceding downloads, hash checks and model runs.

Seven focused tests passed: `ornith_parity`, `ornith_layout_test`, `pool_tasks_test`, `native_grouped_parity`,
`gguf_reader_test`, `gguf_split_test`, and `gguf_geometry_test`. The new expert test checks two CPU tokens at
Ornith's real expert dimensions against independent ggml dot products. Its maximum CPU relative L2 error was
about 6.76e-7; the maximum GPU projection error was about 3.98e-7; dequantized values matched exactly. See
[the numerical output](ornith-expert-parity.txt). The layout test includes ten acceptance/rejection cases.
Tokenizer extraction passed 16 round-trip checks for each model.

The real Python API server also passed a localhost test with the Bartowski configuration: `/health` reported
an 8192-token loaded model, `/v1/models` returned `ornith-1.5-122b-bartowski-q3-k-s`, and
`/v1/chat/completions` returned `391` with `finish_reason: stop`. It ran offline in a separate 56 GiB scope and
was stopped after the request. The final GPU check reported 1 MiB used, 0% utilization, and no LlamaSwap models.
The setup helper's valid-file, wrong-size, wrong-hash and missing-file checks also passed.

The first full-model attempt exposed the missing IQ1_S block-size entry, which was fixed before these passing
runs. A Docker cgroup update also removed GPU access from the build container; inference validation therefore
used the compiled binary on the host with explicit CUDA library paths and systemd memory limits.

HIP and SYCL were not built or tested. Vision, MTP generation, concurrent serving, a full 8K-token prompt and
full-model logit parity against llama.cpp were not tested. The GGML comparison above covers expert arithmetic,
not an assertion of identical complete model logits. The compatibility path uses grouped prompt kernels;
optimized mixed-format MMQ was not implemented in that compatibility build.

The [machine-readable record](ornith-122b.json) contains prompts, output token IDs, outputs, engine protocol
statistics, resource samples and the tested executable's SHA-256. See [setup instructions](../ORNITH.md).
