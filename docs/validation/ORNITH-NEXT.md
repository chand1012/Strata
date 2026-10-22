# Ornith: memory, scheduling and MTP optimization validation

Measured starting 2026-10-21 against `c9ea9f892c3c0b73586a1aae412c34c19d3c261a`. RTX 3090 Ti 24 GiB, i7-10700 (8 physical cores), 64 GB installed RAM; CUDA 13.0.48, GCC 13.3, sm_86, portable AVX2 and pinned ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`. CUDA was built and exercised. HIP, SYCL, Windows and other GPUs were not tested.

All controls are opt-in. The setup helper retains its defaults. This is a targeted follow-up to [the decode report](ORNITH-DECODE.md), not a fresh comparison against Qwen or a broad quality benchmark.

## Implemented changes

- Omit unused MTP weights/state/cache and shrink CPU merge buffers to the maximum hybrid/verification row count. With a 4K chunk, the device CPU-output buffer falls from 384 to 12 MiB; pinned merge storage also shrinks. Released VRAM goes to expert residency.
- Lend prompt-only MMQ and attention workspaces to the expert cache. Track overwritten slots for both streamed prompts and 9–128-row hybrid prompts; restore before decode/MTP uses them. At 4K this lends another 768 MiB. Refills count toward response time and inter-token latency.
- Capture the independent GPU prefix of each eligible main layer, leaving CPU completion before the combine. CUDA external event nodes preserve the host routing dependency. `--no-graph` disables this path.
- Copy missed experts on a separate stream with an explicit consumption dependency. An optional per-layer cost model chooses the offload quota within the allocated bound.
- Rank replacements globally by routing-score gain times observed CPU cost per byte. This changes replacement priority, **not** the uniform main-layer slot allocation. Timing-dependent placement policies reject MTP.
- Add exact AVX2 Q3_K rows sharing unpacked weights across up to four activations, plus singleton IQ down rows. Q3 mode 2 leaves singleton work on ggml. Neither kernel is enabled just because it exists.
- Compute only surviving MTP K/V during prompt processing, cap draft-layer residency separately, and support a reduced 3072-wide Q8 draft head. Main-model verification still uses the complete vocabulary.
- Record per-layer routing/CPU diagnostics without additional device synchronization, and record token-arrival gaps, CPU/GPU clocks and resources in the benchmark harness.

## Method and preliminary screens

Each process unloaded LlamaSwap, verified no running model, and required three GPU-idle samples. Runs used a 56 GiB cgroup with swap disabled. Model loads and one warm-up request were excluded. Timed requests used greedy decoding, fresh 4,096/4,097-token synthetic prose prompts, 8K context, 4K chunks and seven pool workers. Builds and GPU tests ran serially.

One-repeat screens selected candidates for further testing, not final speed claims. The initial Unsloth baseline produced about 17.2 tokens/s whereas the telemetry-only run reached about 20.1 with identical outputs; that difference must not be attributed to trimming or telemetry. Host power policy was unchanged and clocks were recorded. The final comparison reloads both arms in reversed-order blocks.

| Quant | Screen | Prompt tokens/s (4096 / 4097) | Generation tokens/s (4096 / 4097) |
| --- | --- | ---: | ---: |
| bartowski | base | 1047.9 / 1030.6 | 17.75 / 17.41 |
| bartowski | candidate-adaptive | 1046.0 / 1031.6 | 18.62 / 18.97 |
| bartowski | candidate-async | 1047.0 / 1032.4 | 18.46 / 18.53 |
| bartowski | candidate-cost | 1047.4 / 1032.0 | 19.36 / 19.69 |
| bartowski | candidate-cpu | 1045.6 / 1029.9 | 17.75 / 17.81 |
| bartowski | candidate-graph | 1046.1 / 1030.0 | 18.02 / 18.05 |
| bartowski | candidate-workspace | 1018.5 / 1027.9 | 19.55 / 19.22 |
| bartowski | compact | 1046.5 / 1030.9 | 17.99 / 18.09 |
| bartowski | telemetry | 1047.8 / 1030.3 | 17.76 / 17.68 |
| bartowski | trim | 1047.2 / 1031.1 | 18.12 / 17.95 |
| unsloth | base | 1231.0 / 1234.8 | 17.27 / 17.15 |
| unsloth | candidate-adaptive | 1213.3 / 1239.4 | 22.76 / 22.46 |
| unsloth | candidate-async | 1213.3 / 1240.4 | 21.35 / 21.31 |
| unsloth | candidate-cost | 1224.2 / 1239.1 | 23.31 / 23.26 |
| unsloth | candidate-cpu | 1212.7 / 1239.7 | 20.81 / 20.72 |
| unsloth | candidate-graph | 1213.6 / 1239.5 | 20.65 / 20.62 |
| unsloth | candidate-workspace | 1202.4 / 1239.5 | 20.89 / 20.70 |
| unsloth | compact | 1213.5 / 1238.8 | 20.61 / 20.60 |
| unsloth | exact | 1213.4 / 1240.7 | 20.83 / 20.73 |
| unsloth | telemetry | 1233.9 / 1235.5 | 20.22 / 20.03 |
| unsloth | trim | 1220.0 / 1239.5 | 20.35 / 20.33 |

See the [machine-readable evidence](ornith-next.json) for every selection decision, repeats, exact inputs/outputs, commands, flags, binary hashes, resource samples, logs and failures. Workspace rechecks, when present, use three repeats per input length.

## Final repeated comparison

Two independently loaded blocks use opposite arm order, with three repeats per input length in each block: six observations per arm, input length and output limit. The baseline is the preserved starting binary with the previous optimized profile. Values below are medians; generation ranges are minimum–maximum. The 256-token runs request a 600-word essay; the 1024-token runs request 2000 words to avoid early completion. Both arms receive identical input tokens at each output limit, and the long-run controller requires all 1024 tokens. Prompt reuse is rejected. All staging/refill costs remain in the request.

| Quant | Output limit | Input | Arm | Prompt tokens/s | Generation tokens/s (range) | First token s | Total s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| unsloth | 256 | 4096 | baseline | 1258.4 | 20.03 (19.95–20.18) | 3.257 | 16.039 |
| unsloth | 256 | 4096 | candidate | 1256.5 | 23.09 (23.07–23.46) | 3.262 | 14.352 |
| unsloth | 256 | 4097 | baseline | 1239.3 | 20.05 (20.02–20.06) | 3.308 | 16.074 |
| unsloth | 256 | 4097 | candidate | 1239.6 | 23.26 (23.20–23.39) | 3.307 | 14.314 |
| unsloth | 1024 | 4096 | baseline | 1237.5 | 20.07 (19.31–20.45) | 3.312 | 54.373 |
| unsloth | 1024 | 4096 | candidate | 1235.7 | 24.54 (24.39–24.70) | 3.317 | 45.080 |
| unsloth | 1024 | 4097 | baseline | 1220.8 | 20.06 (19.42–20.51) | 3.358 | 54.407 |
| unsloth | 1024 | 4097 | candidate | 1219.4 | 24.57 (24.51–24.73) | 3.362 | 45.032 |
| bartowski | 256 | 4096 | baseline | 1044.8 | 15.96 (14.97–16.85) | 3.922 | 20.004 |
| bartowski | 256 | 4096 | candidate | 1039.3 | 20.33 (19.45–21.12) | 3.943 | 16.564 |
| bartowski | 256 | 4097 | baseline | 1029.3 | 16.47 (15.10–17.86) | 3.982 | 19.614 |
| bartowski | 256 | 4097 | candidate | 1028.2 | 20.39 (19.60–21.22) | 3.987 | 16.563 |
| bartowski | 1024 | 4096 | baseline | 1028.9 | 16.51 (15.54–17.21) | 3.983 | 66.044 |
| bartowski | 1024 | 4096 | candidate | 1027.0 | 22.80 (22.63–22.84) | 3.990 | 48.924 |
| bartowski | 1024 | 4097 | baseline | 1012.5 | 16.10 (15.53–17.66) | 4.048 | 67.747 |
| bartowski | 1024 | 4097 | candidate | 1014.4 | 22.95 (22.72–23.06) | 4.041 | 48.663 |

Promotion requires at least 2% lower median total response time and no more than 2% lower median prompt throughput in every input/output-length combination. Numerical and functional results were reviewed separately; these checks are not a proof of unchanged model quality.

- unsloth: performance gate **True**; median generation gains across the four combinations 15.3%–22.5%; worst prompt ratio 0.9985; worst response ratio 0.8948. The local optimized profile was updated with a backup of its predecessor.
- bartowski: performance gate **True**; median generation gains across the four combinations 23.8%–42.6%; worst prompt ratio 0.9947; worst response ratio 0.8444. The local optimized profile was updated with a backup of its predecessor.

The selected profiles enable trimming, compact merge buffers, asynchronous copies, adaptive offload capped at four experts, and cost-based replacement. Both use cache loans; Bartowski also lends prompt workspaces. Split graphs and the new CPU kernels remain disabled: neither passed the speed-selection screen. MTP also remains disabled. All 96 final timed requests generated their requested token count, used identical inputs between arms, and used no swap. Resource monitoring reported no errors. Sampled GPU core clocks stayed at 1800 MHz; CPU policy was unchanged, and run-to-run variability remains visible above.

The built targets were `strata` and `ornith_q3_test`; the CPU test was then run directly. Exact build flags are in `build-config.json` inside the evidence. To repeat a timing run, write the corresponding exported command/environment configuration to a JSON file and run:

```sh
python tools/bench_ornith.py --command CONFIG.json --model MODEL.gguf \
  --template chat_template.jinja --output NEW_DIRECTORY \
  --lengths 4096,4097 --repeats 3 --new-tokens 256 --workloads prose
```

For the long run use `--new-tokens 1024 --prose-words 2000`. Use a fresh output directory for every process, and preserve both reversed-order blocks.


## Memory and latency

| Quant | Output limit | Arm | Active peak RSS GiB | Active peak anon + shared GiB | Peak GPU MiB | Peak scope GiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| unsloth | 256 | baseline | 56.85 | 43.03 | 23740 | 56.00 |
| unsloth | 256 | candidate | 57.52 | 40.23 | 23794 | 56.00 |
| unsloth | 1024 | baseline | 57.23 | 43.04 | 23740 | 56.00 |
| unsloth | 1024 | candidate | 57.48 | 40.23 | 23794 | 56.00 |
| bartowski | 256 | baseline | 56.64 | 49.88 | 23924 | 56.00 |
| bartowski | 256 | candidate | 58.10 | 48.21 | 23784 | 56.00 |
| bartowski | 1024 | baseline | 58.13 | 49.88 | 23924 | 56.00 |
| bartowski | 1024 | candidate | 57.23 | 48.20 | 23784 | 56.00 |

| Quant | Output limit | Arm | Worst observed token gap s |
| --- | ---: | --- | ---: |
| unsloth | 256 | baseline | 0.227 |
| unsloth | 256 | candidate | 0.226 |
| unsloth | 1024 | baseline | 0.237 |
| unsloth | 1024 | candidate | 0.224 |
| bartowski | 256 | baseline | 0.077 |
| bartowski | 256 | candidate | 0.364 |
| bartowski | 1024 | baseline | 0.078 |
| bartowski | 1024 | candidate | 0.358 |

Pinned host allocations appear as shared memory on this driver. RSS includes those allocations and mapped file pages. Scope usage also includes reclaimable file cache and can reach the configured limit during loading; it is not solely model allocation. GPU memory is sampled for the entire process. Pooled p95 and maximum token-arrival gaps for each input length are in `final-summary.json` inside the evidence; average generation throughput does not describe refill pauses. In particular, Bartowski’s worst observed token gap grows from about 0.078 to 0.364 seconds despite lower total response time. The active anonymous-plus-shared footprint falls by about 2.80 GiB for Unsloth and 1.68 GiB for Bartowski. These reductions do not imply that total RSS falls, since mapped file residency varies.

## MTP screens

| Quant | Configuration | Input | Prompt tokens/s | Generation tokens/s | Accepted / offered |
| --- | --- | ---: | ---: | ---: | ---: |
| unsloth | mtp-serial | 4096 | 1221.1 | 21.41 | 0 / 0 |
| unsloth | mtp-serial | 4097 | 1238.8 | 21.29 | 0 / 0 |
| unsloth | mtp-full | 4096 | 1220.7 | 19.63 | 101 / 150 |
| unsloth | mtp-full | 4097 | 1237.9 | 19.40 | 99 / 149 |
| unsloth | mtp-vocab | 4096 | 1219.7 | 19.87 | 102 / 149 |
| unsloth | mtp-vocab | 4097 | 1236.6 | 19.50 | 99 / 149 |
| unsloth | mtp-two | 4096 | 1219.8 | 17.17 | 129 / 242 |
| unsloth | mtp-two | 4097 | 1236.0 | 16.41 | 124 / 251 |
| bartowski | mtp-serial | 4096 | 1021.5 | 19.36 | 0 / 0 |
| bartowski | mtp-serial | 4097 | 1029.0 | 19.26 | 0 / 0 |
| bartowski | mtp-full | 4096 | 1020.8 | 17.97 | 93 / 157 |
| bartowski | mtp-full | 4097 | 1027.5 | 18.26 | 97 / 153 |
| bartowski | mtp-vocab | 4096 | 1019.8 | 18.13 | 93 / 157 |
| bartowski | mtp-vocab | 4097 | 1027.3 | 18.57 | 97 / 153 |
| bartowski | mtp-two | 4096 | 1020.6 | 16.01 | 119 / 260 |
| bartowski | mtp-two | 4097 | 1027.4 | 15.85 | 118 / 264 |

`mtp-full` means full **draft vocabulary**, with K/V-only prefill; `mtp-vocab` uses the smaller vocabulary and `mtp-two` adds a second draft. MTP controls retain the same MTP weights, draft cap and main-cache capacity as their speculative counterparts. Fixed-placement runs separately compare full MTP prefill with K/V-only prefill. Reduced-vocabulary runs use 131,072 draft tokens and full-vocabulary main verification. Selection also compares against the best non-MTP profile, which can omit draft resources.

## Correctness and limits

- unsloth: all new flags off full logits byte-identical: **True**. Selected-profile teacher-forced positions against the preserved starting binary/profile: 128; mean/max KL 0.0363767/0.987488; argmax agreement 97.66%; forced-continuation mean NLL 0.0820639 → 0.096282. This continuation came from the baseline and is not held-out perplexity. Arithmetic, Python AST, prose/retrieval and 1/2/3-token-limit smoke run passed: **True**.
- bartowski: all new flags off full logits byte-identical: **True**. Selected-profile teacher-forced positions against the preserved starting binary/profile: 128; mean/max KL 0.025128/0.659807; argmax agreement 96.88%; forced-continuation mean NLL 0.101277 → 0.0770844. This continuation came from the baseline and is not held-out perplexity. Arithmetic, Python AST, prose/retrieval and 1/2/3-token-limit smoke run passed: **True**.
- Both MTP functional smoke runs also passed arithmetic, Python AST, retrieval and output-limit checks. MTP remained slower in the screened settings and stays off in the recommended profiles.
- Exact CPU row tests cover Q3_K gate/up/down at widths 1024/3072 and groups 1–8, zero activations and row bounds, plus five singleton IQ down formats against ggml. The combined IQ down/MTP arithmetic dispatch check also passed; a suspected interaction was not reproduced and required no engine change. Microbenchmark data are cache-resident and are not whole-model speed claims.
- Fixed-residency token comparisons exercise arithmetic-preserving controls with 128-, 129- and 4097-token prompts. MTP comparisons check complete generated token sequences against matched serial controls.
- CUDA memcheck runs cover both quants with workspace borrowing, graph execution, transfer overlap and timing-dependent policies; separate MTP runs exercise draft/verification and the smaller draft head. Exact commands and error summaries are included.
- These are limited numerical and functional checks. They do not establish held-out perplexity, multilingual quality, tool-use reliability or equivalence to the unquantized model. Timing-dependent CPU/GPU placement can change near-tie greedy choices through rounding.

## Selected configurations

### unsloth: `candidate-cost`

```json
{
  "command": [
    "$WORK/build-ornith/strata",
    "--native",
    "$WORK/Ornith-1.5-122B-UD-IQ3_XXS.gguf",
    "--max-context",
    "8192",
    "--chunk",
    "4096",
    "--mtp",
    "0",
    "--pool-workers",
    "7",
    "--pool-tasks",
    "0",
    "--serve"
  ],
  "env": {
    "STRATA_ORNITH": "1",
    "STRATA_ORNITH_MMQ": "1",
    "STRATA_MMQ_KQ": "1",
    "STRATA_ORNITH_DECODE": "3",
    "STRATA_ORNITH_REBALANCE": "8",
    "STRATA_ORNITH_PCIE": "4",
    "STRATA_ORNITH_CACHE_LOAN": "1",
    "STRATA_ORNITH_PIN_HOST": "1",
    "STRATA_ORNITH_TRIM_MTP": "1",
    "STRATA_ORNITH_COMPACT_CPU": "1",
    "STRATA_ORNITH_PCIE_ASYNC": "1",
    "STRATA_ORNITH_PCIE_ADAPTIVE": "1",
    "STRATA_ORNITH_CACHE_COST": "1"
  }
}
```

### bartowski: `recheck-workspace`

```json
{
  "command": [
    "$WORK/build-ornith/strata",
    "--native",
    "$WORK/bartowski-q3-k-s/Ornith-1.5-122B-Q3_K_S-00001-of-00002.gguf",
    "--max-context",
    "8192",
    "--chunk",
    "4096",
    "--mtp",
    "0",
    "--pool-workers",
    "7",
    "--pool-tasks",
    "0",
    "--serve"
  ],
  "env": {
    "STRATA_ORNITH": "1",
    "STRATA_ORNITH_MMQ": "1",
    "STRATA_MMQ_KQ": "1",
    "STRATA_ORNITH_DECODE": "3",
    "STRATA_ORNITH_REBALANCE": "8",
    "STRATA_ORNITH_PIN_HOST": "1",
    "STRATA_ORNITH_TRIM_MTP": "1",
    "STRATA_ORNITH_COMPACT_CPU": "1",
    "STRATA_ORNITH_PCIE": "4",
    "STRATA_ORNITH_PCIE_ASYNC": "1",
    "STRATA_ORNITH_PCIE_ADAPTIVE": "1",
    "STRATA_ORNITH_CACHE_COST": "1",
    "STRATA_ORNITH_CACHE_LOAN": "1",
    "STRATA_ORNITH_WORKSPACE_LOAN": "1"
  }
}
```
