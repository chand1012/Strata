# Ornith MTP validation — RTX 3090 Ti

Measured on 2026-10-21, starting from commit `78798402e6198823f5e44a9ce87f9237efe7b145`.
Hardware, model revisions and dependency pin are unchanged from the
[preceding performance validation](ORNITH-PERFORMANCE.md): RTX 3090 Ti 24 GiB,
i7-10700 and 64 GB installed RAM. CUDA was built and exercised; HIP and SYCL were not.

## Repeated performance results

4,096 input tokens, 256 output tokens, five repetitions, medians below. Each engine load
has an excluded warm-up request. Requests are greedy and uncached. Models run serially in
56 GiB scopes with swap disabled; LlamaSwap is unloaded and three idle GPU samples are
required before every model run. The primary speed prompt uses synthetic reference notes.
Separate prose, code and retrieval requests are included in the screening evidence.

“Previous profile” means the settings selected in the preceding optimization step, on the
new engine with all new MTP controls off. “Matched serial” uses the MTP profile with zero
drafts, holding its memory allocation, CPU arithmetic, offload and cache policy constant.

| Quant | Profile | Prompt tokens/s | Generation tokens/s | First token (s) | Total response (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Unsloth | Previous profile | 1261.12 | 18.05 | 3.25 | 17.43 |
| Unsloth | Matched serial | 1261.59 | 17.99 | 3.25 | 17.48 |
| Unsloth | 1 MTP draft | 1186.28 | 17.07 | 3.45 | 18.45 |
| Bartowski | Previous profile | 1043.91 | 16.93 | 3.93 | 19.05 |
| Bartowski | Matched serial | 1046.29 | 15.65 | 3.92 | 20.28 |
| Bartowski | 1 MTP draft | 1019.39 | 15.57 | 4.02 | 20.46 |

Unsloth: MTP generation was -5.1% relative to matched serial and -5.4% relative to the previous profile. Total response time changed by +5.9% relative to the previous profile.

A no-MTP profile had the lowest measured median response time; enabling MTP is not recommended as a speed improvement from this test.

Bartowski: MTP generation was -0.5% relative to matched serial and -8.0% relative to the previous profile. Total response time changed by +7.4% relative to the previous profile.

A no-MTP profile had the lowest measured median response time; enabling MTP is not recommended as a speed improvement from this test.

Individual measurements, ranges, generated outputs, exact input IDs, build hashes and failed
screens are retained in the [measurement record](ornith-mtp.json). No speed improvement is
claimed for sampled requests: this engine only uses MTP for greedy single-request decoding.
The earlier differently quantized Qwen control is not rerun here; MTP does not by itself
establish performance parity with that different architecture and cache allocation.

## Tested MTP profile

Common settings: 8K context, 4,096-token prompt chunks, seven pool workers,
`STRATA_ORNITH=1`, `STRATA_ORNITH_MMQ=1`, `STRATA_MMQ_KQ=1`,
`STRATA_ORNITH_DECODE=3`, and `STRATA_ORNITH_REBALANCE=8`.
Sampling uses temperature zero, repetition penalty one, and frequency/presence penalties zero.
The additional controls are:

- `STRATA_IQ_MT_MIN=1`: use consistent CPU expert arithmetic regardless of token-group size.
- `STRATA_ORNITH_MTP_COMPACT=1`: size prompt buffers separately; preserve capacity with MTP off for matched controls.
- `STRATA_ORNITH_MTP_CACHE=1`: keep draft-layer slots fixed, isolate the main cache replacement budget,
  and align committed state with serial decoding at output limits and end tokens.
- Unsloth: `STRATA_ORNITH_PCIE=2` and `STRATA_ORNITH_PCIE_ROWS=1`. The latter applies the miss limit per token
  for batches of up to eight rows. Bartowski: PCIe miss offload remains disabled.

These settings belong in the model process environment, not a machine-wide environment.
They are opt-in; the existing setup helper and performance configurations keep MTP off.
CPU arithmetic and placement differ from the previous profile, so parity with the matched
serial control does not imply identical text to the previous profile or to a full-precision model.

Unsloth selected `--mtp 1` for repeated testing. Use two drafts only when its mean response time is at least 5% lower than one draft across the three matched workloads.

Bartowski selected `--mtp 1` for repeated testing. Use two drafts only when its mean response time is at least 5% lower than one draft across the three matched workloads.

In a one-run 4K prose screen, adding Q4_0 prompt MMQ reduced Bartowski MTP prompt processing from 6.05 to 4.02 seconds. This isolates a useful prompt-path fix, but the repeated end-to-end measurements above determine the speed recommendation.

## Draft acceptance

| Quant | Draft depth | Accepted / offered | Acceptance |
| --- | ---: | ---: | ---: |
| Unsloth | 1 | 506 / 744 | 68.0% |
| Bartowski | 1 | 464 / 786 | 59.0% |

Acceptance counts only draft proposals; verification also emits main-model tokens. High acceptance alone does not establish a speedup: drafting, verification and reduced main-expert residency all have costs.

## Memory

Peaks include loading and the repeated benchmark. Anonymous plus shared memory excludes
reclaimable file cache; the cgroup peak includes it and is constrained by the 56 GiB limit.
GPU peaks are sampled once per second. These are observed allocations, not minimum-RAM promises.

| Quant / profile | Anonymous + shared (GiB) | Cgroup peak (GiB) | GPU peak (MiB) | Scope swap |
| --- | ---: | ---: | ---: | ---: |
| Unsloth / Previous | 43.03 | 56.00 | 23768 | 0 |
| Unsloth / MTP | 43.04 | 56.00 | 23802 | 0 |
| Bartowski / Previous | 49.88 | 56.00 | 23924 | 0 |
| Bartowski / MTP | 49.88 | 56.00 | 23924 | 0 |

Unsloth MTP prompt staging falls from 4.78 to 3.44 GiB with the compact layout, saving
about 1.34 GiB. The row-wise PCIe mode reserves additional bounded staging. Bartowski’s
compact prompt staging is 2.91 GiB; its Q4_0 draft experts differ from Unsloth’s Q8_0 draft experts.

## Correctness and regression checks

The earlier diagnostic 1K tests fix 32 expert slots and disable replacement and PCIe offload.
With consistent CPU arithmetic, both quantizations match all 384 generated tokens at each
of one and two drafts: 1,536 candidate tokens in total. Every comparison exercises both
accepted and rejected drafts. These frozen-cache controls are diagnostic settings.

The final 4K normal-cache tests compare complete output sequences for prose, code and
retrieval at both draft depths. The selected depth additionally passes five repeated
256-token prose requests against matched serial decoding. The request sequences exercise
cache replacement, length limits, and reset after earlier requests.

Both selected profiles pass arithmetic, parsed Python-function, prose and long-prompt
key-retrieval smoke checks, plus exact one-, two- and three-token output limits. The code
check parses the square function; generated programs are not executed. These checks establish
operation and tested greedy parity, not broad language-model accuracy or held-out perplexity.

The Q4_0 draft-expert format now uses the opt-in prompt MMQ path. This removes the
Bartowski draft layer’s CPU prompt fallback. The matching kernel library uses ABI version 3.
All 96 matrix cases across 12 formats pass the [CUDA numerical check](ornith-mtp-kernels.txt).

The maximum Unsloth verification batch is exercised under CUDA Compute Sanitizer with seven drafts,
a PCIe miss limit of eight per token, and one resident expert slot. The warm-up verifies eight rows.
The [memory-check log](ornith-mtp-sanitizer.txt) reports zero errors. This diagnostic is not a speed run.

With all performance optimization flags off, each quant’s 31,784,960 captured float values are byte-identical
to the preserved pre-optimization reference. Hashes and commands are in the measurement record.

## Failures that informed the fixes

- CPU expert dispatch originally changed arithmetic between single-token and multi-token groups.
  Fixed-placement MTP diverged until `STRATA_IQ_MT_MIN=1` was set on both sides.
- Draft routing originally participated in the main cache’s replacement budget. Normal-cache
  output still diverged after the arithmetic control; the new cache policy isolates draft activity.
- A capped request could commit its final emitted draft token, while serial decoding leaves that
  token unprocessed. This changed the following request’s cache statistics. The bounded draft
  budget and end-token handling prevent those extra committed rows.
- The original offer counter counted requested depth even when a cache boundary shortened the
  window. It now counts actual proposals.

Interrupted follow-up screens are marked as controller interruptions, not model crashes.
Vision, sampled speculation, concurrent MTP serving and broader quality evaluation were not tested.

## Reproduce

Build as described in the preceding performance report. Use `tools/bench_ornith.py` with a
command JSON containing the profile above, `--lengths 4096 --new-tokens 256 --repeats 5
--workloads prose`, and a fresh output directory. Repeat with the same settings and `--mtp 0`.
Then compare the two directories:

```sh
python tools/compare_ornith_mtp.py /path/to/serial-results /path/to/mtp-results
```

The comparator requires identical input token sequences, completed uncached runs and identical
output tokens. It reports first differences and whether speculation, acceptance and rejection
were exercised. Keep the JSON inputs, outputs, resource samples and engine logs for review.
