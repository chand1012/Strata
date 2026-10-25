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
