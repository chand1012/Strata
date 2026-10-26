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

## Head to head (same GGUF, same machine, 2026-10-25)
| test | llama.cpp best tuned | Strata (defaults) | |
|---|---|---|---|
| pp512 | 181.5 (`--n-cpu-moe 31 -t 8 -ub 512`) | 213.4 / 226.6 (two runs) | Strata +18-25% |
| pp2048 | 532.8 (`--n-cpu-moe 34 -ub 2048 -b 2048`) | 569.5 | Strata +7% |
| pp4096 | 875.6 (`--n-cpu-moe 36 -ub 4096 -b 4096`) | 985.9 (`STRATA_Q36_CHUNK=4096`) | Strata +13% |
| tg128 (bench) | 14.45 (`--n-cpu-moe 31 -t 16`) | 15.9 (after pp2048), 19.4 (after pp512) | Strata +10-34% |
| 256-token chat decode | 14.1 (llama-server); 9.97 with MTP | 18.1; 19.3 with `--mtp 1` (setup's default) | Strata +34% |
Strata commands: `strata-q36 --model <shard1> --max-context 8192 --bench-pp N --bench-tg 128` (3rd of 3 runs);
decode: `--tokens <code prompt> --n 256 [--mtp 1]`. llama.cpp: `llama-bench -m <shard1> -ngl 99 -fa 1 ...` as listed.
Short prompts were first lost (pp512 153.4): every chunk streamed all non-resident experts over PCIe 3.0. Fix:
STRATA_Q36_STREAM_CPU (Ornith default `auto` = 0.5 for chunks <= 1024 rows): half of the non-resident experts, the
least used, are not copied; the CPU pool computes their entries while PCIe brings the rest. pp512 sweep: 0.2 183,
0.35 213, 0.5 251, 0.65 218; at pp2048 0.35 gave 562 (off: 570), hence off above 1024 rows.
Correctness of that path (1228-token prompt, 512-row chunks, 16 forced steps vs llama.cpp): KL 0.054 with it, 0.046
without, top-1 94% / 100%. --n-cpu-moe 27-30 did not fit in VRAM for llama.cpp.

## Speed round 2 (2026-10-26)
Baseline = branch at d91b6ac0 (Ornith defaults). Commands: `strata-q36 --model <shard1> --max-context 8192
--bench-pp N --bench-tg 128` (3rd of 3 runs; pp4096 with the default prompt chunk) and the 256-token greedy decode on
the code prompt (`--tokens <ids> --n 256 [--mtp 1]`). Script: ~/work/fullbench.sh.
| config | pp512 | tg128 (after pp512) | pp2048 | tg128 (after pp2048) | pp4096 | chat 256 | chat 256 --mtp 1 |
|---|---|---|---|---|---|---|---|
| baseline | 226.3 | 19.4 | 569.3 | 15.8 | 573.8 | 17.3 | 19.4 (82%) |
| STRATA_Q36_STAGE_SPLIT=1 (staging sized for the IQ3_S layers; blk.47 and MTP streamed in 2 parts): 59 instead of 53 slots | 221.6 | 20.2 | 578.4 | 15.2 | 584.3 | 18.1 | 19.7 (75%) |
| + STRATA_Q36_FETCH_CACHE=1 (decode keeps fetched experts in the idle staging buffers, LRU) | 233.6 | 21.5 | 580.8 | 19.0 | 583.6 | 20.5 | 21.2 (69%) |

Idea 1 (free VRAM for expert slots), details:
- Stage split: logits bitwise identical with it on and off (same 50 fixed slots, a 1228-token streamed prompt, 16 forced
  steps). Kept: Ornith default.
- Fetch cache (reuses the staging buffers during decode instead of freeing them). Fetch share sweep with the cache,
  teacher-forced 256-token decode (`dec2.sh`): 0.3 17.8, 0.4 19.8, 0.5 20.4 (twice), 0.6 19.6, 0.7 17.5, 1.0 14.0;
  without the cache 18.2. Output changes (more experts on the GPU): ornith_kl.py mean KL 0.0195 vs floor 0.0381, PASS;
  long streamed prompt KL 0.040 (before 0.046). Kept: Ornith default.
- Token embedding table in host memory (STRATA_Q36_EMBD_CPU=1): logits bitwise identical, 63 instead of 59 slots, but
  decode slower: 17.1 / 16.9 vs 18.2 / 18.0, and 17.3 at the same 59 slots. Kept opt-in, off.
- KV cache precision: skipped. Measured the whole KV cost at the launch context: 32K 52 slots 19.8 tok/s, 8K 59 slots
  20.4 tok/s; fp16 KV would win back at most half of that 3% and touches the attention kernels and session snapshots.
- Default context: left at 32K for launch (a user choice; it costs 3% decode against 8K, measured above).

Idea 2 (slot placement), teacher-forced 256-token decode on the code prompt (`dec2.sh`), current defaults = 20.5 tok/s:
- Hot-expert profile (new opt-in STRATA_Q36_PROFILE / STRATA_Q36_PROFILE_OUT / STRATA_Q36_PROFILE_W): a profile from
  three other prompts (story, the French one, a 1228-token summary), 256 tokens each. Start slots from it and keep it
  as a floor under the routing scores: W=1 19.2 / 19.1, W=4 17.9 / 17.8, W=0.25 20.0, W=0 (start slots only) 20.4.
  Routing follows the current prompt more than a fixed profile, and the post-prompt rebalance (2048 swaps) already
  places the slots. Kept opt-in, off.
- Per-layer slot counts by traffic: from the same profile, the share of routed entries the slots cover at the same
  total VRAM is 53.9% with 59 slots in every layer and 54.0% with the best per-layer split (48..77 slots). Not built.

Idea 3 (overlap PCIe and CPU in decode), teacher-forced 256-token decode (`dec2.sh`), defaults before = 20.4-20.5:
- Next-layer prefetch (opt-in STRATA_Q36_PREFETCH=K): layer L+1's router on layer L's post-attention residual predicts
  79% of L+1's top-8 (offline, from the llama.cpp dumps; 82% from the post-MoE residual, 90% recall with 12
  guesses). The prefetched experts are used (29,985 copied, 30,647 cache hits) but decode is slower because PCIe is
  already busy: K=2 15.7, K=4 13.6, K=8 12.4; with less reactive fetch: F=0.4 K=1 20.0, F=0.25 K=2 17.6, F=0.01 K=3
  16.5. Kept opt-in, off.
- Adaptive fetch share (STRATA_Q36_FETCH_ADAPT=1): per layer, up a step when its copies were done before the CPU pool
  finished, down a step otherwise. Step 0.02: 21.4; step 0.005: 21.6; starting from 0.3: 21.5 (fixed 0.5: 20.5).
  Output changes (which experts run on the GPU): ornith_kl.py mean KL 0.0197 vs floor 0.0381, PASS. Kept: Ornith
  default, step 0.005.
| + STRATA_Q36_FETCH_ADAPT=1 | 229.8 | 24.8 | 580.7 | 20.1 | 583.4 | 21.6 | 20.9 (64%) |
