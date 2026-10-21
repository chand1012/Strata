# Ornith decode optimization — RTX 3090 Ti

Measured on 2026-10-21, starting from `5bf202a9d764f45d3fd06547165c252cebaa91a5`.
Hardware, models and dependency pins are unchanged from [the previous report](ORNITH-MTP.md):
RTX 3090 Ti 24 GiB, i7-10700, 64 GB installed RAM, CUDA 13.0.48, GCC 13.3, sm_86,
portable AVX2 and the pinned ggml build. CUDA compilation and execution were tested.
HIP and SYCL were not built or exercised. This is a targeted comparison, not the repository's
standard benchmark or a new comparison with Qwen.

## What changed

`STRATA_ORNITH_PIN_HOST=1` applies the existing CPU-pool affinity mechanism to Ornith's
inference thread. The pool already reserves a physical core for the host and pins its workers,
but this engine path had not pinned the host. The native engine does so in `SessionLoopScratch`.
On this system the unpinned host could use CPUs 0–15 while workers were pinned to CPUs 1–7.
The opt-in pins the host once to its reserved CPU, restores its prior affinity on exit, and
restores the input reader's inherited mask. It does not change expert arithmetic or worker count.

`STRATA_ORNITH_CACHE_LOAN=1` lets prompt staging share a contiguous GPU expert arena.
Long prompts stream through its prefix using the existing MMQ kernels. Generation restores
that prefix from host memory before reading cached experts. Short prompt tails restore only
the resident experts they route to, avoiding a full-cache transfer before the first output token.
All these transfers are included in prompt, decode and total-response measurements.
The bulk refill occurs before processing the first generated token, so it adds latency between
the first and second output tokens. This is not a latency-free memory saving; inter-token gaps
were not measured separately. Refill counts and elapsed transfer times are recorded in the logs.

This keeps the 4K chunk and streamed-prompt kernels unchanged. Small hybrid tails can change
CPU/GPU placement. It adds cached experts without allocating separate prompt buffers during generation. The flag requires a partial hybrid
cache, MMQ for every main layer, MTP off, one text session, and at least eight hybrid rows.
The existing defaults and setup helper are unchanged. The optional `--pool-tasks N` exposes
the CPU pool's existing task-granularity setting; zero retains its previous default.

## Selected profiles and host-affinity comparison

The selected 256-output-token profile uses host pinning plus cache reuse for Unsloth, and
host pinning alone for Bartowski. Existing profiles and setup defaults are unchanged. All
new controls are opt-in. For shorter responses, Unsloth's pin-only profile avoids the bulk
refill pause; a cache-reuse win on 256 output tokens does not establish a win on short answers.

The following complete runs have five repetitions per input length; values are medians with
minimum–maximum generation ranges. Requests are greedy and uncached, with 256 output tokens
and an excluded warm-up request per model load. Requests alternate
between 4,096 and 4,097 input tokens. Both pinned variants use the same inputs and settings
as their corresponding unpinned controls except for host affinity. Pin-only keeps the old
cache capacity; pin + reuse uses the larger expert arena. PCIe limits remain two for Unsloth
and zero for Bartowski.

| Quant | Input tokens | Profile | Prompt tokens/s | Generation tokens/s (range) | First token (s) | Total response (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Unsloth | 4096 | Original | 1260.70 | 17.73 (17.67–19.74) | 3.251 | 17.688 |
| Unsloth | 4096 | Pin only | 1260.66 | 19.48 (19.40–19.52) | 3.251 | 16.393 |
| Unsloth | 4096 | Pin + reuse | 1259.11 | 19.99 (19.95–20.17) | 3.255 | 16.063 |
| Unsloth | 4097 | Original | 1239.11 | 18.01 (17.99–19.65) | 3.308 | 17.523 |
| Unsloth | 4097 | Pin only | 1242.27 | 19.65 (19.59–19.69) | 3.300 | 16.328 |
| Unsloth | 4097 | Pin + reuse | 1239.41 | 20.04 (20.02–20.12) | 3.308 | 16.079 |
| Bartowski | 4096 | Original | 1044.18 | 15.28 (15.22–15.59) | 3.925 | 20.678 |
| Bartowski | 4096 | Pin only | 1045.30 | 17.54 (17.39–17.73) | 3.920 | 18.517 |
| Bartowski | 4096 | Pin + reuse | 1040.47 | 18.23 (15.98–18.77) | 3.939 | 17.976 |
| Bartowski | 4097 | Original | 1027.87 | 15.50 (15.28–16.65) | 3.988 | 20.516 |
| Bartowski | 4097 | Pin only | 1030.28 | 17.65 (17.58–17.78) | 3.978 | 18.483 |
| Bartowski | 4097 | Pin + reuse | 1029.32 | 16.80 (15.95–18.58) | 3.982 | 19.221 |

The original host could run on CPUs 0–15 while workers were pinned to CPUs 1–7. Live masks
confirm that the new mode keeps the host on CPU 0, workers on CPUs 1–7, and the input reader
on the original broad mask. The Linux affinity regression test passes. Across both quantizations,
host pinning preserves all 10,240 compared output tokens: 2,560 per quant and cache configuration.
These exact comparisons separate thread placement from changed expert arithmetic.

CPU clocks were not fixed. Point-in-time readings showed roughly 4.6–4.7 GHz in one Bartowski
control and about 4.0 GHz during some slower combined-cache cases, with normal temperatures.
The powersave / balance_performance policy and frequency limits were not changed. All repetitions,
including the slow ones, are retained. Timing differences cannot all be attributed to the code
changes; the larger Bartowski cache is not recommended from these variable results. Host-only
pinning was more consistent in the tested sequences. These are operational measurements under
the existing host policy, not fixed-frequency kernel benchmarks.

Selection minimizes the sum of median response times for the two input lengths, requiring
prompt throughput and first-token latency within 3% of the original profile. This is a targeted
choice for the tested workload, not an exhaustive optimum. Task granularity and PCIe limits
were screened before host pinning; those screens do not establish their optimum after pinning.
MTP with host pinning was not retested. Generation at 1,024 output tokens and broad workload
performance were not measured in this pass.

Selected-profile memory peaks:

| Quant | Anonymous + shared RAM (GiB) | Cgroup peak (GiB) | GPU peak (MiB) | Swap bytes |
| --- | ---: | ---: | ---: | ---: |
| Unsloth | 43.03 | 56.00 | 23740 | 0 |
| Bartowski | 49.88 | 56.00 | 23924 | 0 |

## Cache reuse with the original host placement

Each row is the median of five uncached, greedy prose requests with 256 output tokens.
Every model load has an excluded warm-up request. The initial screen runs control then
candidate; these repeated runs reverse that order. Individual runs and ranges are retained
in [the measurement record](ornith-decode.json). Repetitions within a load are not independent
process launches. Prompt text uses synthetic repeated reference notes. Larger cache placement can
change greedy continuations through CPU/GPU rounding, so these free-generation timings compare
the complete profiles; they do not isolate cache-hit cost on identical generated token streams.

First-token and response times start when tokenized requests are sent through the engine protocol;
they exclude model loading, chat templating and client tokenization.

Aggregate engine counters include the excluded harness warm-up; speed tables use only measured requests.

All models run serially in 56 GiB scopes with swap disabled. Before each load the harness
unloads LlamaSwap, confirms no model remains, and requires three idle GPU samples. Compilation
and numerical analysis do not overlap timed model runs.

| Quant | Input tokens | Cache reuse | Prompt tokens/s | Generation tokens/s | First token (s) | Total response (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Unsloth | 4096 | Off | 1260.70 | 17.73 | 3.251 | 17.688 |
| Unsloth | 4096 | On | 1259.07 | 18.67 | 3.255 | 16.970 |
| Unsloth | 4097 | Off | 1239.11 | 18.01 | 3.308 | 17.523 |
| Unsloth | 4097 | On | 1237.20 | 18.70 | 3.313 | 17.007 |
| Bartowski | 4096 | Off | 1044.18 | 15.28 | 3.925 | 20.678 |
| Bartowski | 4096 | On | 1043.51 | 18.67 | 3.927 | 17.637 |
| Bartowski | 4097 | Off | 1027.87 | 15.50 | 3.988 | 20.516 |
| Bartowski | 4097 | On | 1028.11 | 18.56 | 3.987 | 17.778 |

The 4,097-token case exercises a full 4K chunk followed by a one-token tail. Separate single-run
screens cover 512 and 7,168 input tokens. These are workload-specific measurements; they do not
establish speed at other contexts, long output lengths, concurrent requests or on other hardware.
MTP remains off: its preceding implementation was correct in tested comparisons but slower.

## One-expert PCIe screen

Five repetitions per input length with cache reuse enabled and the host unpinned. One missed
expert per layer/token did not improve the combined median response time relative to the
existing limits. No one-expert profile was selected. These results precede the affinity fix.

| Quant | Input tokens | One-expert generation tokens/s | Total response (s) |
| --- | ---: | ---: | ---: |
| Unsloth | 4096 | 17.24 | 18.104 |
| Unsloth | 4097 | 17.21 | 18.194 |
| Bartowski | 4096 | 18.32 | 17.930 |
| Bartowski | 4097 | 18.12 | 18.119 |

## CPU task-size screen

One measured 4K/256 request per setting after warm-up. Seven workers plus the host participate;
zero selects the existing three tasks per participant. All tested settings produced identical
output tokens within each quantization. The default remains selected.

| Quant | Default (0) | 12 tasks | 48 tasks | 96 tasks |
| --- | ---: | ---: | ---: | ---: |
| Unsloth | 19.72 | 18.11 | 18.89 | 19.68 |
| Bartowski | 17.73 | 16.66 | 17.45 | 17.45 |

Values are generation tokens/s. These screens use the CPU-control build before the cache-reuse
code was compiled; binary hashes are recorded. Compare settings within these screens, not against
a different request history or an earlier day's best number. The host was unpinned, so
scheduling variability limits these comparisons.

## Memory

Peaks include loading. Anonymous plus shared memory excludes reclaimable file cache; the cgroup
peak includes it. GPU memory is sampled once per second. These are observations, not minimum-RAM
requirements. Exact cache allocations and refill totals are preserved in the engine summaries.

| Quant | Cache reuse | Anonymous + shared (GiB) | Cgroup peak (GiB) | GPU peak (MiB) | Swap bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| Unsloth | Off | 43.03 | 56.00 | 23768 | 0 |
| Unsloth | On | 43.03 | 56.00 | 23740 | 0 |
| Bartowski | Off | 49.88 | 56.00 | 23924 | 0 |
| Bartowski | On | 49.88 | 56.00 | 23754 | 0 |

## Correctness

With cache placement fixed at 51 Unsloth or 43 Bartowski slots per layer, enabling reuse produces
identical complete 64-token outputs for 128-, 512- and 4,097-token prompts, in the same request
sequence. That is 192 compared generated tokens per quant. This checks restoration separately
from the numerical effects of moving more expert work from CPU to GPU.

With all optimization flags off, both quants' 31,784,960 captured float values are byte-identical
to the preserved reference. The CPU pool task and Linux affinity tests pass. Both selected profiles pass the
arithmetic, parsed Python function, key-retrieval and output-limit smoke checks. CUDA Compute
Sanitizer reports zero errors for each quant, with host pinning and cache reuse enabled,
32 fixed expert slots per layer, a diagnostic PCIe limit of eight, 256-token chunks, a 513-token
input and eight output tokens. This exercises streaming, a short tail, and decoding. Its timings
are not speed results.

Increasing GPU residency changes which CPU/GPU arithmetic processes some experts. The following
128-position teacher-forced comparison uses the same input and continuation on the original
and selected performance profiles (pin + reuse for Unsloth, pin only for Bartowski). It is not
held-out perplexity or a broad accuracy evaluation.

| Quant | Mean KL | Maximum KL | Argmax agreement | Control forced NLL | Selected forced NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unsloth | 0.022343 | 0.531790 | 98.44% | 0.108982 | 0.125351 |
| Bartowski | 0.000000 | 0.000000 | 100.00% | 0.078916 | 0.078916 |

The selected Bartowski profile also preserves the complete captured logits byte-for-byte.
Unsloth's larger cache does not: its top-token choices agree at 126 of 128 forced positions.

These checks establish tested operation and numerical comparisons, not equality to full-precision
Ornith or guaranteed accuracy on arbitrary tasks. Full outputs, hashes, commands and checks are
in the measurement record. Memory-check logs: [Unsloth](ornith-decode-unsloth-sanitizer.txt),
[Bartowski](ornith-decode-bartowski-sanitizer.txt).

## Reproduce

Build with the pinned CUDA instructions from [the performance report](ORNITH-PERFORMANCE.md).
Use its 8K context, 4K chunk, seven workers, MTP zero, MMQ and integer-dot kernels, cache replacement
budget eight, and PCIe miss limit two for Unsloth or zero for Bartowski. Keep `--pool-tasks 0`.
Add `STRATA_ORNITH_CACHE_LOAN=1` to a copy of the model process environment for the candidate.
Run `tools/bench_ornith.py` with `--lengths 4096,4097 --new-tokens 256 --repeats 5 --workloads prose`
and a new output directory for each configuration. Reverse order on the follow-up comparison.
For exact restoration parity, add the same `--expert-slots` value to both configurations and
compare the complete outputs with the `compare()` function in `tools/compare_ornith_mtp.py`.
Its command-line MTP-specific speculation requirement does not apply to this serial experiment.

For the selected profiles, add `STRATA_ORNITH_PIN_HOST=1` to both model environments and
`STRATA_ORNITH_CACHE_LOAN=1` only to Unsloth. Local `server-performance-optimized.json` files
contain these tested settings. Separate `server-performance-pinned.json` files use host pinning
without cache reuse. Neither profile enables MTP. The new engine build retains K-quant library
ABI version 3; no CUDA expert kernels changed in this pass.

The task-size, cache/offload, and host-affinity stages were compiled successively. Individual
binary and library hashes are recorded for every run. The final flags-off logit check verifies
that the complete patch preserves the original default outputs; other models and backends
were not used to establish that result.
