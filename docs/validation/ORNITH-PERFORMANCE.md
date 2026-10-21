# Ornith performance validation — RTX 3090 Ti

Measurement date: 2026-10-21. This records the opt-in performance work on the two Ornith 1.5 122B
GGUFs supported by this fork. It does not validate vision, MTP, HIP, SYCL, or other hardware.

## Test system and method

RTX 3090 Ti, 24 GiB VRAM, 400 W power limit; Intel i7-10700 (8 physical cores / 16 threads),
64 GB installed RAM (about 61 GiB usable), PCIe 3.0 x16. CUDA 13.0.48, driver 595.91.07,
GNU 13.3, Release build for sm_86 with portable AVX2 CPU kernels. Compilation used a CUDA development
container; Strata full-model inference ran on the host in systemd scopes. The CPU used its existing
intel_pstate powersave / balance_performance policy; observed inference clocks were around 3.8 GHz.

The baseline is compatibility commit `5d0f4a74752a92fea87f1bbebf07456218ddd011`, based on upstream
`9de04cd44e927a9a3df6bbc0e94fc1be8b844ba4`. The ggml dependency is pinned to
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. The model revisions and hashes are in the setup helper.
The baseline executable and its original MMQ library were preserved before building the candidate.

Each full-model run unloads LlamaSwap, checks that no model is listed as running, and requires three idle
GPU samples. Models run one at a time in a 56 GiB memory-limited scope with swap disabled. Tests use
8,192 context, MTP off, greedy generation, complete chat templates with thinking disabled, an excluded
warm-up request, and uncached input. The harness asserts zero reused tokens on every Strata request.
The primary matrix uses 512, 1,024, 4,096 and 7,168 input tokens and 256 generated tokens, five repeats
per length. Its repeated reference-note filler is synthetic; separate corpus checks address that limitation.

Report prompt processing, generation and time to first token separately. A larger prompt chunk reserves
more workspace and can reduce the number of resident experts, improving prompt speed while hurting
generation. Resource samples are taken once per second. For Strata, anonymous plus shared memory is
reported separately from the cgroup total, which also includes reclaimable file cache. For mmap-based
llama.cpp, file-backed resident weights must also be counted. Neither model's download size is its peak RAM.

## Confirmed optimized results

Five-repeat medians, the same input token IDs as the baseline, and 256 output tokens.

At 4K, Unsloth prompt processing improved **23.8×** and total response time fell from **96.30 to 17.66 seconds**. Generation changed from 17.57 to 18.00 tokens/s.

At 4K, Bartowski prompt processing improved **22.1×** and total response time fell from **103.37 to 19.26 seconds**. Generation changed from 20.61 to 16.90 tokens/s.

| Quant | Input tokens | Prompt tokens/s | Generation tokens/s | First token (s) | Total response (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unsloth | 512 | 144.45 | 20.53 | 3.55 | 16.00 |
| Unsloth | 1,024 | 304.47 | 19.67 | 3.36 | 16.38 |
| Unsloth | 4,096 | 1190.87 | 18.00 | 3.44 | 17.66 |
| Unsloth | 7,168 | 1095.30 | 17.24 | 6.55 | 21.39 |
| Bartowski | 512 | 121.48 | 19.46 | 4.22 | 17.31 |
| Bartowski | 1,024 | 254.02 | 18.68 | 4.03 | 17.73 |
| Bartowski | 4,096 | 996.11 | 16.90 | 4.11 | 19.26 |
| Bartowski | 7,168 | 910.43 | 16.18 | 7.88 | 23.70 |

These settings target the measured prompt-plus-256-token workload. Prompt speed and generation speed are separate tradeoffs; the original configuration remains available. Minima, maxima and individual repeats are in the [measurement record](ornith-performance.json).

### Selected settings

Both use `STRATA_ORNITH=1`, `STRATA_ORNITH_MMQ=1`, `STRATA_MMQ_KQ=1`,
`STRATA_ORNITH_DECODE=3`, and `STRATA_ORNITH_REBALANCE=8`, with 8K context and MTP off.

| Quant | Chunk | Pool workers | PCIe misses per layer |
| --- | ---: | ---: | ---: |
| Unsloth | 4096 | 7 | 2 |
| Bartowski | 4096 | 7 | 0 |

Set `STRATA_ORNITH_PCIE` to the table value; zero disables it. Use the matching engine and MMQ library. These are measured opt-in settings, not new defaults.

The additional five-repeat Bartowski worker check measured 20.92 seconds at fifteen workers and 19.26 seconds at seven for the 4K response. The final configuration uses 7 workers under the stated 5% selection rule.

### Memory

Peaks include startup and the complete repeated matrix. The cgroup peak includes file cache; anonymous plus shared memory better describes Strata’s non-file-cache allocation. It is not a universal minimum-RAM specification. Process RSS was not captured in the first Unsloth baseline. Summed RSS can count shared mappings more than once and uses different accounting from the memory-limited cgroup.

| Quant / path | Anonymous + shared (GiB) | Sampled process RSS (GiB) | Cgroup peak (GiB) | Sampled GPU peak (MiB) | Scope swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unsloth / baseline | 42.58 | Not sampled | 56.00 | 23466 | 0 |
| Unsloth / selected-final | 43.03 | 57.66 | 56.00 | 23766 | 0 |
| Bartowski / baseline | 49.43 | 55.85 | 56.00 | 23462 | 0 |
| Bartowski / selected-final | 49.88 | 56.89 | 56.00 | 23922 | 0 |

### Same-machine references

4K input / 256 output, five-repeat medians. llama.cpp uses the immutable image in the measurement record, eight threads, 36 CPU MoE layers, batch 512 and microbatch 256. This is the previously validated offload setting, not an exhaustive llama.cpp tuning result. All reference requests report zero prompt-cache reuse. Qwen uses a different IQ2_XS quant and a different architecture.

| Runtime / model | Prompt tokens/s | Generation tokens/s | First token (s) | Total response (s) |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp / Unsloth | 114.79 | 16.82 | 35.69 | 50.85 |
| llama.cpp / Bartowski | 96.37 | 18.36 | 42.50 | 56.45 |
| Strata / Qwen IQ2_XS, default tail skipping | 1845.79 | 98.07 | 2.23 | 4.82 |
| Strata / Qwen IQ2_XS, tail skipping off | 1752.22 | 97.48 | 2.35 | 4.96 |

Neither Qwen control uses MTP, suffix drafting or lookup drafting. “Tail skipping off” means all routed experts are retained; it is not a claim of full-precision model accuracy.

### Separate corpus checks

Two repeats each of prose, code and retrieval at 4K, using repository documentation and code rather than repeated filler. These are additional operation/workload checks; no matched baseline corpus speedup is claimed. Code output is capped at 256 tokens and may be incomplete; only the separate square-function smoke test is checked as executable Python syntax.

| Quant | Workload | Generated tokens (range) | Prompt tokens/s | Generation tokens/s |
| --- | --- | ---: | ---: | ---: |
| Unsloth | prose | 256–256 | 1256.69 | 16.38 |
| Unsloth | code | 256–256 | 1254.75 | 16.60 |
| Unsloth | retrieval | 214–225 | 1255.38 | 17.98 |
| Bartowski | prose | 256–256 | 1045.34 | 14.79 |
| Bartowski | code | 256–256 | 1045.81 | 15.44 |
| Bartowski | retrieval | 239–245 | 1044.90 | 16.24 |

### Full-model numerical and smoke checks

Both selected configurations passed arithmetic, parsed Python-function, short-prose and long-prompt key-retrieval smoke checks. Their generated outputs are retained in the measurement record.

The numerical comparison uses a 1,024-token prompt and 127 teacher-forced continuation tokens from baseline generation, producing 128 logit vectors. It compares against the previous Strata build on the same GGUF, not a full-precision model or llama.cpp. KL is in nats. Forced-token NLL is not held-out perplexity.

| Quant | Mean KL | Maximum KL | Top-token agreement | Baseline forced NLL | Optimized forced NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unsloth | 0.029400 | 0.476314 | 95.31% | 0.082674 | 0.108982 |
| Bartowski | 0.012679 | 0.317433 | 97.66% | 0.076932 | 0.078916 |

The optimized path changes numerical results. These checks do not establish broad task accuracy or compare the inherent quality of the two quantizations. Formal quality evaluation and MTP work remain outside steps 1–5.

With all optimization flags off, the new build’s 31,784,960 captured float values were byte-identical to the original build for each quant. This checks the tested prompt and continuation, not every possible input. See the [kernel and sanitizer output](ornith-performance-kernels.txt) for the separate numerical tests.

## Original compatibility baseline

Five-repeat medians; 256 generated tokens, 32-token chunks and seven pool workers plus the host thread.

| Quant | Input tokens | Prompt tokens/s | Generation tokens/s | First token (s) |
| --- | ---: | ---: | ---: | ---: |
| Unsloth UD-IQ3_XXS | 512 | 33.82 | 20.55 | 15.14 |
| Unsloth UD-IQ3_XXS | 1,024 | 46.35 | 19.98 | 22.09 |
| Unsloth UD-IQ3_XXS | 4,096 | 50.12 | 17.57 | 81.73 |
| Unsloth UD-IQ3_XXS | 7,168 | 50.17 | 16.26 | 142.87 |
| Bartowski Q3_K_S | 512 | 47.56 | 22.57 | 10.77 |
| Bartowski Q3_K_S | 1,024 | 47.84 | 22.96 | 21.41 |
| Bartowski Q3_K_S | 4,096 | 45.07 | 20.61 | 90.89 |
| Bartowski Q3_K_S | 7,168 | 43.88 | 19.22 | 163.35 |

## Diagnosis and changes

The original compatibility path uses CPU experts for cache misses during prompt processing. A synchronized
stage diagnostic spent about 77% of prompt time and 71% of decode time waiting for CPU work and merging it.
Those diagnostic timings perturb overlap and are not throughput measurements. An independent 14.94-second
Nsight prompt capture showed GPU kernels or copies active for 3.47 seconds (23.3% of the capture span).
CPU performance-counter sampling was unavailable under the host's existing security policy.

The candidate adds these opt-in paths:

- Grouped MMQ prompt products for IQ1_S, Q2_K and Q3_K, alongside the existing IQ/K kernels. Gate and up
  projections can use different formats and activation layouts. Eligibility is checked per layer.
- Prompt staging sized for eligible active layers. With MTP disabled, Unsloth staging falls from 4.78 to
  2.09 GiB and Bartowski staging to 2.91 GiB. This does not unload the MTP weights.
- Integer-dot expert projections that consume compressed weights directly. Fractional IQ2_XXS and IQ2_S
  scale terms are retained in the new path; existing callers keep their original arithmetic.
- An optional rotating cache-replacement schedule with consistent score aging, and bounded PCIe offload
  of single-token misses. Both are explicitly selectable and leave the original schedule intact when unset.

The final expert microtests measured about 0.074–0.103 ms for the original projections and
0.0166–0.0272 ms for integer-dot projections across the nine tested format triples. These short isolated
timings are not a whole-model speedup: CPU work, transfers, routing and the remaining layers still matter.
Packed-warp and split-reduction experiments did not help and were removed.

## Configuration screening

These are single measured requests after an excluded warm-up, not five-repeat conclusions. All use the
final kernel build. On the original grouped path, 64 / 128 / 256-token chunks achieved 46.3 / 49.1 / 49.9
prompt tokens/s at 1K; increasing the chunk alone did not fix prompt processing. With MMQ, 2,048-token
chunks and rotating cache replacement, 3 / 7 / 11 / 15 pool workers generated at 9.0 / 16.6 / 17.5 / 20.2
tokens/s at 1K. The 15-worker setting therefore received a separate 4K combination check.

The main 4K screens use seven pool workers, 256 output tokens, integer-dot projections, and a rotating
cache replacement budget of eight. PCIe is the maximum number of missed experts sent to the GPU per layer.

| Quant | Chunk | PCIe | Prompt tokens/s | Generation tokens/s | Total response (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unsloth | 2048 | 0 | 628.6 | 15.5 | 23.06 |
| Unsloth | 2048 | 1 | 628.4 | 17.5 | 21.15 |
| Unsloth | 2048 | 2 | 626.6 | 18.9 | 20.07 |
| Unsloth | 4096 | 0 | 1259.6 | 16.9 | 18.43 |
| Bartowski | 2048 | 0 | 525.0 | 17.4 | 22.51 |
| Bartowski | 2048 | 2 | 520.8 | 17.3 | 22.70 |
| Bartowski | 4096 | 0 | 1046.6 | 15.4 | 20.55 |

Selection prefers the smaller chunk and less PCIe offload if within 5% of the fastest screened total
response time. A worker candidate must improve the combined 4K response by at least 5% before replacing
seven workers. This avoids recommending a more expensive setting based on a small screening difference.
The selected settings receive the five-repeat matrix and separate corpus checks.

A separate 64-token diagnostic reached 29.4 generation tokens/s after warming the cache. It is not comparable
to the long-prompt matrix and is not evidence of sustained 30–40 tokens/s on representative long prompts.

## Numerical validation

The expanded expert tests cover nine gate/up/down format triples at Ornith's actual 3,072/1,024 geometry,
with independent ggml references, permuted rows, zero inputs and empty groups. Both the original and integer-dot
paths pass. The integer-dot CUDA Compute Sanitizer run reports zero errors. IQ1_S checks account explicitly for
the Q8_1 stored sum used by the pinned integer-dot arithmetic, rather than loosening the numerical tolerance.

The MMQ test covers 11 quantization types and 88 cases, both projection shapes, varied group sizes, empty
experts, padded strides, offsets and destination leading dimensions. Exactly representable activation values
isolate matrix and layout correctness from activation-rounding error. Maximum observed relative L2 error was
0.00148 against an independent CPU calculation (limit 0.005). This is a kernel test, not a language-quality score.
A preconverted FP16 microbenchmark can win for large single-expert groups, but no full-model FP16 advantage
was established or enabled.

## Comparison limits

The upstream README table uses an RTX 5070 / Ryzen 7600 and lists IQ3_XXS at 62 generation and
1,750 prompt tokens/s; its 3090 figure of 100–140 generation tokens/s is an estimate. The
[length-specific table](../DETAILS.md#speed-measured) lists IQ3_XXS prompt rates of 482 at 1K,
1,007 at 4K and 1,745 at 32K. The headline prompt rate is not a 4K result. These are
upstream claims, not measurements reproduced by this work. The accompanying DETAILS benchmark explicitly
has MTP speculative decoding enabled. This campaign keeps MTP off for both Ornith and the local Qwen
control; implementing and validating Ornith speculation belongs to the later plan step and was not done here.

The installed Qwen control uses IQ2_XS, not the advertised IQ3 quant. Downloading the latter would require
about 75.8 GB, exceeding the available disk space without deleting working models. Its results must therefore
be labelled as a differently quantized control. Ornith has 3,072-wide embeddings, 1,024-wide experts and eight
active experts per layer; the installed Qwen has 2,560, 640 and ten respectively. Ornith consequently performs
1.536 times as much routed expert matrix work per layer. This ratio excludes all other model work and does
not predict total runtime or imply that matching Qwen's advertised speed is achievable on this system.

The local Qwen control initially rejected `--spec 0`: its native-pack serving path requires an execution
window of at least two. The corrected run uses `--spec 2`, no MTP model, `--suffix-draft 0`, and
`--lookup-chain 0`. All 20 completed requests reported zero drafts and zero reused tokens. The default
Qwen policy also enables `STRATA_ROUTE_TAIL_SKIP=7`, which skips some missed experts and changes outputs.
That run is retained separately; an additional control sets `STRATA_ROUTE_TAIL_SKIP=0`.

The control's cache allocation is another material difference: 14,135 expert slots / 18.95 GiB, with
4.18 GiB borrowed temporarily for prompts. The 2,048-token Unsloth screen has 61 of 256 experts per layer /
9.84 GiB of cache, plus persistent 2.09 GiB prompt staging and the engine's other allocations. This helps
explain why a faster GPU expert kernel alone does not close the generation gap. Reclaiming prompt allocations
for decode would require additional cache-placement/lifecycle work and has not been implemented here.

## Reproduce the primary benchmark

Build the engine and `libstrata_kq.so` from this branch. The measured build used the following
CMake settings (paths below are placeholders for a fresh build):

```sh
cmake -S /path/to/Strata -B /path/to/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_PORTABLE=ON \
  -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_GGML_DIR=/path/to/pinned-llama.cpp
cmake --build /path/to/build --target strata strata_kq
```

Generate model/tokenizer files using
`tools/ornith_setup.py`, then create a command JSON with `command` (an argument array ending in `--serve`)
and `env` (environment overrides). Use the exact model path, chunk, pool-worker count and flags shown in the
measurement record. Run each configuration alone:

```sh
python tools/bench_ornith.py --command /path/to/command.json \
  --model /path/to/model.gguf --template /path/to/tokenizer/chat_template.jinja \
  --output /path/to/new-results-directory --workloads prose \
  --lengths 512,1024,4096,7168 --new-tokens 256 --repeats 5
```

The output directory must not already exist. Keep `inputs.json`, `results.json`, `engine.log` and
`resources.jsonl`. Profiling flags must be absent for timing runs. For independent prompt material, add
`--corpus /path/to/text.txt`; the corpus must contain enough tokens for the requested lengths. The tool
unloads LlamaSwap at the configured URL before checking GPU idleness.
