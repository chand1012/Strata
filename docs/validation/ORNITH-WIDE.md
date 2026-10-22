# Ornith: 3072-wide kernels and per-layer cache allocation

The new kernels produced small generation gains with essentially unchanged prompt throughput and memory use. Neither candidate met the promotion threshold on every workload, so the existing runtime profiles remain unchanged. Cache redistribution and larger PCIe ceilings did not help the tested workloads.

Measured starting 2026-10-22 on RTX 3090 Ti 24 GiB, i7-10700 and 64 GB installed RAM. The baseline is `925951ecf8d479bca02500622ef9d4f7f8c2c2dc`, using the profiles selected in [the previous optimization report](ORNITH-NEXT.md). CUDA 13.0.48, GCC 13.3, sm_86 and portable AVX2; the pinned ggml revision is unchanged. CUDA was built and executed. HIP, SYCL, Windows and other hardware were not tested.

## Changes

- `STRATA_ORNITH_WIDE` is a bit mask: 1 extends unpacked Q8 rows to 96 quantization blocks; 2 enables the shared gate/up/SwiGLU/quantization fusion at width 3072; 4 extends fused MoE output, normalization and quantization to width 3072. Zero retains the old dispatch. Prompt GEMMs are unchanged. Verification scratch buffers now cover the actual embedding width.
- Tensor-header inspection found Q8 shared gate/up pairs in 17 of 48 Unsloth main layers and all 48 Bartowski main layers. The remaining Unsloth shared gate/up weights are Q5_K, so the Q8 fusion does not cover them. Fused MoE output applies to both quants.
- `STRATA_ORNITH_CACHE_PROFILE=1` records committed decode routing counts and singleton CPU expert costs per main layer. `tools/ornith_cache_slots.py` uses these measurements to propose fixed per-layer counts, accounting for byte strides and aligned tails. Add-one smoothing, a floor at half the original count, and a ceiling at twice that count temper calibration overfitting.
- `STRATA_ORNITH_LAYER_SLOTS` supplies 48 explicit counts. Runtime validation rejects counts outside 1–255, incompatible configurations and allocations above the automatic byte budget. Boundaries remain fixed for the process; the existing expert replacement policy still runs. This is offline calibration, not dynamic repartitioning or router prediction.

## Method

Each model process first unloads LlamaSwap and checks three idle GPU samples. Runs use a 56 GiB cgroup with swap disabled. Model loading and one benchmark warm-up are excluded; prompt processing, transfers and cache restoration remain inside each timed request. Builds and tests are serialized.

Calibration uses 1024-token code and retrieval prompts. Speed screens use separate 4096/4097-token prose prompts with 256 output tokens and two repeats. The old binary and the rebuilt binary with new options off are both screened. Kernels are tested individually and together; promising combinations are then checked separately. Additional screens raise the existing adaptive PCIe offload ceiling from four to six or eight experts without changing routed expert selection.

Final runs use two independent loads per arm/output limit in reversed arm order, with two repeats per input length in each load: four observations per arm, input length and output limit. The 256-token task requests 600 words; the 1024-token task requests 2000 words and must reach all 1024 tokens. These synthetic workloads do not represent every application.

## Final timings

| Quant | Output limit | Input | Arm | Prompt tokens/s | Generation tokens/s (min–max) | First token s | Total s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| unsloth | 256 | 4096 | baseline | 1240.0 | 23.02 (22.96–23.14) | 3.306 | 14.410 |
| unsloth | 256 | 4096 | candidate | 1240.0 | 23.64 (23.46–23.80) | 3.306 | 14.134 |
| unsloth | 256 | 4097 | baseline | 1239.2 | 23.29 (23.14–23.36) | 3.308 | 14.297 |
| unsloth | 256 | 4097 | candidate | 1239.5 | 23.66 (23.50–23.70) | 3.307 | 14.128 |
| unsloth | 1024 | 4096 | baseline | 1225.1 | 24.42 (24.23–24.73) | 3.346 | 45.243 |
| unsloth | 1024 | 4096 | candidate | 1224.7 | 24.90 (24.85–25.07) | 3.347 | 44.465 |
| unsloth | 1024 | 4097 | baseline | 1219.1 | 24.61 (24.50–24.69) | 3.363 | 44.980 |
| unsloth | 1024 | 4097 | candidate | 1219.7 | 24.99 (24.90–25.09) | 3.361 | 44.335 |
| bartowski | 256 | 4096 | baseline | 1035.0 | 21.04 (20.99–21.14) | 3.960 | 16.109 |
| bartowski | 256 | 4096 | candidate | 1035.7 | 21.34 (21.30–21.43) | 3.957 | 15.941 |
| bartowski | 256 | 4097 | baseline | 1029.8 | 21.14 (20.96–21.32) | 3.980 | 16.098 |
| bartowski | 256 | 4097 | candidate | 1030.2 | 21.41 (21.28–21.54) | 3.979 | 15.934 |
| bartowski | 1024 | 4096 | baseline | 1020.0 | 21.92 (20.86–22.87) | 4.018 | 50.825 |
| bartowski | 1024 | 4096 | candidate | 1021.9 | 23.13 (22.98–23.16) | 4.010 | 48.282 |
| bartowski | 1024 | 4097 | baseline | 1012.1 | 22.60 (21.25–23.00) | 4.050 | 49.371 |
| bartowski | 1024 | 4097 | candidate | 1014.7 | 23.11 (22.97–23.41) | 4.039 | 48.350 |

### Run-to-run variation

| Quant | Output | Block | Baseline generation tokens/s | Candidate generation tokens/s |
| --- | ---: | ---: | ---: | ---: |
| unsloth | 256 | 0 | 23.24 | 23.68 |
| unsloth | 256 | 1 | 23.07 | 23.58 |
| unsloth | 1024 | 0 | 24.59 | 24.99 |
| unsloth | 1024 | 1 | 24.52 | 24.92 |
| bartowski | 256 | 0 | 21.09 | 21.36 |
| bartowski | 256 | 1 | 21.07 | 21.38 |
| bartowski | 1024 | 0 | 21.13 | 23.15 |
| bartowski | 1024 | 1 | 22.85 | 23.07 |

Bartowski’s long-output baseline changed substantially between independent loads. Its first-block apparent gain is not a stable isolated kernel speedup. Adaptive output sequences differ and change subsequent expert routing.

Promotion requires at least 2% lower median total response time and no more than 2% lower median prompt throughput in every input/output-length combination. Correctness checks are reviewed separately.

## Screening results

Ratios compare each candidate with the rebuilt binary with new options off. These short screens select candidates; the repeated final timings determine promotion.

| Quant | Candidate | Response-time ratio | Worst prompt-throughput ratio |
| --- | --- | ---: | ---: |
| unsloth | wide0 | 1.0000 | 1.0000 |
| unsloth | wide1 | 1.0016 | 1.0002 |
| unsloth | wide2 | 1.0019 | 0.9997 |
| unsloth | wide4 | 0.9887 | 1.0009 |
| unsloth | wide7 | 0.9861 | 1.0013 |
| unsloth | allocated | 1.0038 | 1.0005 |
| unsloth | pcie6 | 1.0173 | 1.0006 |
| unsloth | pcie8 | 1.6032 | 0.9882 |
| bartowski | wide0 | 1.0000 | 1.0000 |
| bartowski | wide1 | 0.9989 | 1.0001 |
| bartowski | wide2 | 1.0029 | 1.0003 |
| bartowski | wide4 | 0.9898 | 1.0003 |
| bartowski | wide7 | 0.9909 | 1.0002 |
| bartowski | allocated | 1.0011 | 1.0002 |
| bartowski | pcie6 | 1.0045 | 0.9998 |
| bartowski | pcie8 | 1.7786 | 0.9847 |

- unsloth: selected candidate `wide7`; performance gate **False**; worst prompt ratio 0.9997, worst response ratio 0.9881. Local optimized profile retained.
- bartowski: selected candidate `wide4`; performance gate **False**; worst prompt ratio 1.0003, worst response ratio 0.9898. Local optimized profile retained.

## Memory and latency

| Quant | Output limit | Arm | Peak active RSS GiB | Peak active anon + shared GiB | Peak GPU MiB | Worst token gap s |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| unsloth | 256 | baseline | 57.54 | 40.23 | 23794 | 0.226 |
| unsloth | 256 | candidate | 57.77 | 40.23 | 23794 | 0.226 |
| unsloth | 1024 | baseline | 58.06 | 40.24 | 23794 | 0.224 |
| unsloth | 1024 | candidate | 57.69 | 40.23 | 23794 | 0.224 |
| bartowski | 256 | baseline | 57.99 | 48.21 | 23784 | 0.360 |
| bartowski | 256 | candidate | 57.35 | 48.21 | 23784 | 0.358 |
| bartowski | 1024 | baseline | 56.47 | 48.21 | 23784 | 0.366 |
| bartowski | 1024 | candidate | 55.85 | 48.21 | 23784 | 0.357 |

Pinned memory appears as shared memory on this driver. RSS also includes mapped file pages; the cgroup includes reclaimable cache and can reach its 56 GiB limit during loading. Neither RSS nor scope usage is solely model allocation. Exact resource samples and pooled token-gap percentiles are retained.

## Correctness and limits

- unsloth: wide-kernel GPU reference comparisons 33960, mismatches 0; memcheck passed **True**. Unequal-cache memcheck (75% of calibrated counts to reserve sanitizer VRAM) passed **True**. Fixed-placement complete token comparison passed **True**. Legacy default full logits byte-identical **True**.
- unsloth: selected-profile teacher-forced argmax agreement 96.09% over 128 positions; mean/max KL 0.0148199/0.390171; mean forced-continuation NLL 0.106092 → 0.106128. Functional smoke passed **True**.
- bartowski: wide-kernel GPU reference comparisons 38420, mismatches 0; memcheck passed **True**. Unequal-cache memcheck (75% of calibrated counts to reserve sanitizer VRAM) passed **True**. Fixed-placement complete token comparison passed **True**. Legacy default full logits byte-identical **True**.
- bartowski: selected-profile teacher-forced argmax agreement 96.88% over 128 positions; mean/max KL 0.0320184/2.16736; mean forced-continuation NLL 0.0930392 → 0.104069. Functional smoke passed **True**.
- Kernel verification covers batches 1–8 and 129/513-token prompt cases. Default and fixed-placement checks separate arithmetic identity from timing-dependent placement. New residency and scheduling timings can still change CPU/GPU rounding and greedy tokens. Every matched adaptive pair in the first timing block diverged in generated token IDs (first differences at positions 12–90). The speed results therefore measure end-to-end generation with different subsequent routing workloads; they do not isolate kernel throughput or establish unchanged answer quality.
- Zero-count and over-budget allocations were rejected with the expected diagnostics for both quantizations (four runtime checks). Unequal-cache reference comparisons totaled 30,279 for Unsloth and 39,929 for Bartowski, with zero mismatches.
- With explicit allocations, use the printed `calibrated main-layer slots` vector for actual layer capacities; the legacy uniform-slot summary retains the automatic reference count.
- CPU-only allocator tests cover heterogeneous byte strides, capacity bounds, marginal preference, unchanged allocation with a full floor, and invalid profiles. The first synthetic fixture used an incorrect marginal-benefit expectation; correcting the fixture required no allocator change.
- The first Unsloth sanitizer run completed with zero kernel mismatches but did not capture the sanitizer’s stdout summary. Its memory result is recorded as inconclusive. The capture was corrected and the checks rerun; both initial and repeated records are retained.
- The first full-allocation Unsloth memcheck was rejected by the automatic cache-budget guard because the sanitizer reduced free VRAM. No inference ran in that attempt. Unequal-cache memchecks use each calibrated count scaled to 75%; full allocations were exercised by the normal timing screens. Both configurations and the rejected attempt are retained.
- The first fixed-placement comparison invocation passed a results file where the helper required a directory. The model runs had completed successfully; the harness was corrected and those same results were compared. The original error is retained.
- The forced continuation comes from baseline generation. These checks do not establish held-out perplexity, broad factual accuracy, multilingual quality or tool-use reliability. A calibrated allocation may not generalize to other workloads, contexts or quantizations.

## Calibration and reproduction

Run a representative workload with `STRATA_ORNITH_CACHE_PROFILE=1`, save stderr, then propose an allocation:

```sh
python tools/ornith_cache_slots.py calibration.log --output allocation.json
python tools/test_ornith_cache_slots.py
```

Apply the generated environment value only after testing on separate workloads. Keep the original configuration for comparison. To repeat a benchmark with the exported command/environment JSON:

```sh
python tools/bench_ornith.py --command CONFIG.json --model MODEL.gguf \
  --template chat_template.jinja --output NEW_DIRECTORY \
  --lengths 4096,4097 --repeats 2 --new-tokens 256 --workloads prose
```

For long responses add `--prose-words 2000` and use `--new-tokens 1024`. The [raw evidence](ornith-wide.json) contains screen results, exact commands and inputs, calibration counts, binary/source hashes, failed or inconclusive checks, numerical checks and resource samples. `$WORK` replaces the local test directory; substitute an actual path before use.

## Candidate configurations

### unsloth

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
    "LD_LIBRARY_PATH": "$WORK/build-ornith:$WORK/runtime",
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
    "STRATA_ORNITH_CACHE_COST": "1",
    "STRATA_ORNITH_WIDE": "7"
  }
}
```

### bartowski

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
    "LD_LIBRARY_PATH": "$WORK/build-ornith:$WORK/runtime",
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
    "STRATA_ORNITH_WORKSPACE_LOAN": "1",
    "STRATA_ORNITH_WIDE": "4"
  }
}
```
