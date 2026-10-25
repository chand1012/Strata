# Ornith-1.5-122B IQ3_XS: llama.cpp baseline vs Strata

Machine: RTX 3090 Ti 24 GB, i7-10700 (8C/16T, AVX2), 61 GB RAM, Linux, CUDA 13.0 (docker nvidia/cuda:13.0.0-devel).
File: bartowski Ornith-1.5-122B-IQ3_XS (2 shards, sha256 checked). llama.cpp build f3269ef (2026-10-24).

## llama.cpp (llama-bench, -ngl 99 -fa 1)
| config | pp512 | pp2048 | pp4096 | tg128 |
|---|---|---|---|---|
| --n-cpu-moe 31 -t 8 -ub 512 | 181.5 | 178.5 | | 14.03 |
| --n-cpu-moe 31 -t 16 -ub 512 | 178.7 | 178.1 | | 14.45 |
| --n-cpu-moe 34 -t 8 -ub 2048 -b 2048 | | 532.8 | | |
| --n-cpu-moe 36 -t 8 -ub 4096 -b 4096 | | | 875.6 | |
| --n-cpu-moe 40 -t 8 -ub 4096 -b 4096 | | | 826.9 | |

llama-server, greedy, 256 tokens: --n-cpu-moe 31 -t 8: 14.1 / 13.7 tok/s. With MTP
(`--spec-type draft-mtp --spec-draft-n-max 2`, --n-cpu-moe 36 to fit): 9.97 / 9.93 tok/s (acceptance 0.58-0.60).
Best llama.cpp: decode 14.45 tok/s, prefill 532.8 (pp2048) / 875.6 (pp4096).

Command: `llama-bench -m <shard1> -ngl 99 -fa 1 -t 8 -p 2048 -n 0 -b 2048 -ub 2048 --n-cpu-moe 34`

## Correctness vs llama.cpp (teacher-forced, full-vocabulary logits)
3 prompts (fr / code / story), 64 steps each: llama.cpp's greedy continuation fed to both engines, KL(llama || engine)
per step over the whole vocabulary. Reference: `tools/qwen35/q36_llama_dump --cpu-moe 1` (llama.cpp f3269ef).
| engine vs reference | fr | code | story | mean | top-1 agree |
|---|---|---|---|---|---|
| llama.cpp `--ngl 14` (noise floor) | 0.037 | 0.009 | 0.050 | 0.032 | 94.8% |
| Strata (default, MMQ prompt) | 0.035 | 0.016 | 0.022 | 0.024 | 95.3% |
| Strata (STRATA_Q36_NO_MMQ=1) | 0.034 | 0.022 | 0.019 | 0.025 | 94.8% |
| Strata before the pool fix | 0.124 | 0.134 | 0.342 | 0.200 | 87.5% |

The bug found: the CPU pool's native split buffers were `float ff[MAXT][640]` (Flash-Next's width); Ornith's
expert width is 1024, so an expert serving several tokens wrote one token's row over the next. Fixed with
`kNativeMaxFF = 1024` (wider experts are refused, not overrun).

## Strata optimizations, one at a time (strata-q36 --bench-pp N --bench-tg 128, 3rd of 3 runs; decode = 256 greedy
tokens on the code prompt)
| step | pp2048 | pp4096 | tg128 | 256-token decode |
|---|---|---|---|---|
| first run (no MMQ: the Q4_0 MTP experts turned it off) | 118 | | 13.5 | |
| Q4_0 MTP experts re-encoded -> MMQ + streamed prompt chunks | 523.5 | | 12.2 | 12.0 |
| STRATA_Q8_PACKED=0 (VRAM for 53 instead of 44 slots/layer) | 523.1 | | 11.2 | 14.9 |
| + STRATA_Q36_CHUNK=4096 | | 1047 | | |
| + STRATA_Q36_FETCH=0.5 (half the missed experts over PCIe beside the CPU pool) | 526.2 | | | 18.1 |
| + STRATA_Q36_STREAM_RUNS=1 (prompt stream copies only non-resident experts) | 573.2 | | | |
| all three (now the Ornith defaults) | 568.1 | 985.8 | 15.8 | |
| defaults + --mtp 1 | | | | 19.3 (82% accepted) |
| Fetch sweep (decode): 0.3 16.3, 0.4 16.1, 0.5 18.1, 0.6 16.1, 0.7 15.0, 0.8 13.2 | | | | |

Not helping here: --cpu-threads 16 (7-10 tok/s), 6 or 7 threads (same as 8), STRATA_IQ3S_MT1, huge pages (13.2-13.3,
same as without), MTP 2/3 drafts (11.1 / 9.4).
Measured limits: PCIe 3.0 x16 12.9 GB/s pinned H2D; the CPU pool's IQ3_S expert rows 20-24 GB/s on 8 threads
(kq_cpu_bench --h 3072 --ff 1024). The prompt stream waits 2.84 s of a 3.87 s pp2048 for copies (STRATA_Q36_STREAM_PROF).
Tiers: VRAM slots -> pinned RAM (all 48.2 GiB of experts fit in 61 GB RAM, so there is no SSD tier for this file).
