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
is the helper's initial default; it is not a recommendation from the model's author. No vision or MTP support
is claimed by this addition.

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
* Prompt processing uses the same grouped compatibility path. MMQ is disabled for these layouts, so it cannot
  interpret mixed gate/up matrices as one uniformly quantized matrix. Optimized mixed-format MMQ and fused
  decode kernels are future performance work, not prerequisites for the compatibility path.

This path is intended for CUDA. HIP and SYCL have not been validated. The normal fused expert paths do not
select the compatibility kernel unless the Ornith flag is set. CUDA graph capture and CPU/GPU routing still use
the existing engine scheduler.

## Validation

Build and run the focused checks on an otherwise idle GPU:

```sh
cmake --build build --target ornith_parity pool_tasks_test native_grouped_parity \
  gguf_reader_test gguf_split_test gguf_geometry_test ornith_layout_test -j 6
ctest --test-dir build -R '^(ornith_parity|pool_tasks_test|native_grouped_parity|gguf_reader_test|gguf_split_test|gguf_geometry_test|ornith_layout_test)$' --output-on-failure
```

`ornith_parity` checks the actual 3072/1024 expert geometry, IQ1_S/Q2_K, Q3_K, mixed Q4_K/Q5_K and IQ down
projections against independent ggml calculations. It checks tensor byte sizes, CPU pool results, GPU decoder
values, GPU matrix products and empty-group behavior. Numerical tests do not replace complete model tests.
See the [validation record](validation/ORNITH-122B.md) for measured results and remaining limitations.
