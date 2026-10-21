# Ornith 1.5 122B (experimental CUDA support)

This fork adds an explicit `STRATA_ORNITH=1` compatibility path for these GGUFs:

| Model file | Download | Packaging |
| --- | --- | --- |
| Unsloth UD-IQ3_XXS | 49.52 GB | One file; mixed IQ/K tensor types |
| Bartowski Q3_K_S | 57.48 GB | Two shards; includes a Q4_K gate / Q5_K up layer |

Both are ordinary GGUF files. Unsloth's dynamic quantization uses different quantization types across
projections; it is not a separate container format. The setup helper pins the revisions, byte sizes and SHA-256
hashes of all three files. Keep both Bartowski shards together and pass the first shard to the engine.

The initial configuration targets one NVIDIA GPU with 24 GB VRAM and a host with 64 GB RAM. It uses 8K context,
one request at a time, text only, eight CPU threads, a 32-token prompt chunk and MTP disabled. Greedy sampling
is the helper's initial default; it is not a recommendation from the model's author. Experimental greedy MTP
has a separate [validation report](validation/ORNITH-MTP.md). Vision has not been validated.

## Build and prepare

Compile this branch, rather than downloading an upstream release binary. For an RTX 3090 / 3090 Ti:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_NATIVE_EXPERTS=ON
cmake --build build --target strata -j 6
```

Install the repository's Python requirements in a virtual environment. Use the setup helper explicitly:

```sh
python tools/ornith_setup.py --quant unsloth-iq3-xxs \
  --models /path/to/models --engine build/strata --output /path/to/ornith-config
```

For Bartowski use `--quant bartowski-q3-k-s`. Add `--download` to fetch missing files. Existing files are
verified and never overwritten. The helper extracts the tokenizer and chat template from the selected GGUF,
and prints a `serve/server.py --engine strata --config ...` command. The resulting server binds to localhost. The normal model
catalog and existing configurations are unchanged. The engine and its CUDA runtime libraries must be available
to the server process, as with other source builds. Pass `--lib-dir /path/to/cuda/libraries` to the helper
when they are not on the system library search path. This option can be repeated.

For direct engine use, set `STRATA_ORNITH=1` and pass the GGUF with `--native`. The environment flag is carried
in the generated server configuration. Unload other GPU models before starting; this helper does not manage
LlamaSwap or terminate other applications.

## Implementation

* The GGUF size table includes IQ1_S (256 values / 50 bytes).
* The opt-in layout check accepts `[3072, 1]` for the shared expert gate vector. Other tensor shapes retain strict
  validation. The Ornith loader checks the 3072-dimensional embedding, 1024-dimensional experts and 256 experts.
* Expert blobs track the up projection's size, type and offset independently from the gate. CPU mixed-format
  execution uses ggml's separate dot products and requires a common activation type.
* The CPU pool provides space for 1024 hidden values and their Q8_K activation blocks, with an explicit capacity
  check before using that storage. Legacy model dimensions and arithmetic remain unchanged.
* The compatibility GPU kernel supports the quantization types in these files, including IQ1_S, Q2_K and Q3_K.
  It expands one 256-value weight block in shared memory at a time and consumes Q8_1 activations. Entire experts
  stay compressed in RAM and VRAM.
* The default prompt path remains the grouped compatibility path. Opt-in MMQ handles gate and up independently
  when their types differ, including their activation-quantization layouts. Unsupported layers retain the grouped
  path without disabling MMQ for the other layers. The Q4_0 MTP experts in the Bartowski file are included.
* The optional integer-dot decoder reads compressed weights directly. IQ2_XXS and IQ2_S keep fractional scale
  terms in this path. Existing callers retain their original arithmetic.
* With MMQ enabled, streaming buffers cover the largest eligible active layer. A disabled MTP block does not
  enlarge those buffers unless the compact MTP control is enabled for matched comparisons; its weights remain loaded.

This path is intended for CUDA. HIP and SYCL have not been validated. The normal fused expert paths do not
select the compatibility kernel unless the Ornith flag is set. CUDA graph capture and CPU/GPU routing still use
the existing engine scheduler.

## Performance controls

These controls are opt-in and specific to the Ornith compatibility path. The build must include the matching
`libstrata_kq.so` beside the engine; this change uses version 3 of that library's internal ABI.

| Environment variable | Behavior |
| --- | --- |
| `STRATA_ORNITH_MMQ=1` and `STRATA_MMQ_KQ=1` | Enable grouped int8 prompt products and expert streaming. Chunks larger than the hybrid-row threshold (128 by default) use streaming. |
| `STRATA_ORNITH_DECODE=3` | Enable direct integer-dot expert projections. An unset value retains the compatibility kernel. |
| `STRATA_ORNITH_REBALANCE=N` | Rotate the starting layer of decode cache replacement and age all layer scores. Allow up to N replacements every 32 tokens, capped at 512. Unset or zero preserves the original scheduler. |
| `STRATA_ORNITH_PCIE=N` | Send at most N missed experts per layer to the GPU during single-token decoding, capped at 8. Requires pinned host weights and reserves a small staging buffer. Unset or zero leaves misses on the CPU. |
| `STRATA_ORNITH_MTP_COMPACT=1` | Size the two prompt buffers separately for the isolated MTP pass. Retain the same capacity with MTP off for matched controls. |
| `STRATA_ORNITH_MTP_CACHE=1` | Keep draft-layer slots fixed and exclude draft routing from the main cache replacement budget. Align speculative state with serial decoding at output limits and end tokens. |
| `STRATA_ORNITH_PCIE_ROWS=1` | Apply the PCIe miss limit per token in batches of up to eight rows, including MTP verification. Reserves extra bounded staging; requires `STRATA_ORNITH_PCIE` above zero. |
| `STRATA_ORNITH_PROFILE=1` | Print synchronized stage timings and streamed byte counts. This diagnostic changes overlap and must not be used for speed claims. |
| `STRATA_ORNITH_TRACE=1` | Start a CUDA profiler capture after engine warm-up. Use with Nsight Systems' CUDA profiler API capture range. |

`--chunk` and `--pool-workers` remain normal engine arguments. Larger chunks amortize streaming transfers but
reserve more VRAM and leave fewer resident experts for generation. Increasing PCIe offload or cache-copy budgets
can make generation slower. See the [performance validation](validation/ORNITH-PERFORMANCE.md) for measured
settings, comparisons, and limits on the tested 3090 Ti / 64 GB system.

The measured 8K-context performance configurations use `--chunk 4096`, MTP off, and the MMQ and integer-dot
flags above, plus `STRATA_ORNITH_REBALANCE=8`:

| Quant | `--pool-workers` | `STRATA_ORNITH_PCIE` |
| --- | ---: | ---: |
| Unsloth UD-IQ3_XXS | 7 | 2 |
| Bartowski Q3_K_S | 7 | 0 |

Apply these arguments and environment values to a copy of the generated server configuration. They target
the measured prompt-plus-256-token workload. In particular, Bartowski's faster prompt processing comes with
slower generation than its original configuration. The setup helper keeps the original conservative defaults.
See the validation record for full timings, memory usage, numerical differences and comparison limits.

The benchmark utility `tools/bench_ornith.py` runs exact-length chat prompts through the engine protocol. It accepts
an engine command/environment JSON, the GGUF and its chat template, and a new output directory. It unloads
LlamaSwap, checks GPU idleness, enforces a 56 GiB no-swap scope, excludes a warm-up request, and rejects prompt
reuse. It records inputs, outputs, binary/library hashes, first-token latency, engine timings, and resource samples.
It requires Linux, a systemd user session, `nvidia-smi`, the repository Python dependencies, and LlamaSwap (URL can
be set with `--llamaswap`). Its default filler is synthetic repeated text; `--corpus` supplies other prompt material.

## Experimental MTP

The [MTP report](validation/ORNITH-MTP.md) compares one and two drafts with matched serial decoding and the
previous performance profile. It includes repeated speed and memory measurements, exact output comparisons,
and the failures that required fixes. MTP is not automatically faster on this machine.

Use the complete tested profile from that report, including `STRATA_IQ_MT_MIN=1` on both the serial and MTP
controls. This keeps CPU expert arithmetic consistent across verification batch sizes. Keep sampling greedy
(temperature zero, repetition penalty one, frequency and presence penalties zero); requests that need
sampling logits use serial decoding. The setup helper and existing performance profiles
continue to disable MTP. `tools/compare_ornith_mtp.py` compares complete output token sequences from benchmark
runs with identical input IDs.

## Validation

Build and run the focused checks on an otherwise idle GPU:

```sh
cmake --build build --target ornith_parity ornith_mmq_test pool_tasks_test native_grouped_parity \
  gguf_reader_test gguf_split_test gguf_geometry_test ornith_layout_test -j 6
ctest --test-dir build -R '^(ornith_parity|ornith_integer_parity|ornith_mmq_test|pool_tasks_test|native_grouped_parity|gguf_reader_test|gguf_split_test|gguf_geometry_test|ornith_layout_test)$' --output-on-failure
```

`ornith_parity` checks the actual 3072/1024 expert geometry, IQ1_S/Q2_K, Q3_K, mixed Q4_K/Q5_K and IQ down
projections against independent ggml calculations. It checks tensor byte sizes, CPU pool results, GPU decoder
values, GPU matrix products and empty-group behavior. Numerical tests do not replace complete model tests.
See the [validation record](validation/ORNITH-122B.md) for measured results and remaining limitations.
