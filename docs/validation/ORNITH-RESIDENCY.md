# Ornith expert residency and replacement

This experiment starts from `32fdc35753d9b8145515dfe7d5f3433318f2307b` on an RTX 3090 Ti (24 GiB), i7-10700 and 64 GB installed RAM. All new controls are opt-in. The previous optimized profiles are the controls; kernel-wide flags remain off.

## Implementation

`STRATA_ORNITH_CACHE_PRIOR=/path/model.prior` seeds each main layer's resident slots with its most frequently used experts. The same counts contribute a persistent prior to replacement ranking. Live routing counts still decay and adapt to the request. `STRATA_ORNITH_CACHE_PRIOR_TOKENS` sets the prior strength in equivalent routing tokens per layer (default 256, allowed 0–65536). Zero still seeds the initial identities but adds no subsequent prior weight.

Generate the file from a separate calibration run with `STRATA_ORNITH_CACHE_PROFILE=1`:

```sh
python3 tools/ornith_cache_prior.py calibration-engine.log --output model.prior
```

The reader validates dimensions, all per-layer expert strides, finite nonnegative integral counts, positive layer totals and end-of-file. It rejects truncated or incompatible files. It does not hash the model's weights: keep the profile with its exact model/quant and record the calibration corpus. A profile from one task is not evidence that its ranking generalizes to another.

`STRATA_ORNITH_REPLACE_ASYNC=1` queues decode replacements on a separate stream in layer order. Selection still uses the existing cost policy. Each layer waits for its weights and pinned mapping snapshot to finish before reading them. The previous forward finishes before any slot is overwritten. Prompt entry drains pending copies before prompt staging can borrow the same arena. Prompt-end replacement remains synchronous; only decode replacements are pipelined.

`STRATA_ORNITH_DEFER_LOAN_REPLACE=1` avoids copying replacement weights into cache bytes already invalidated by prompt staging. Their new expert identities are published normally, and the existing refill restores them before use. This does not omit refill time from decode or total response time.

These controls require Ornith hybrid placement, the cost policy, MTP off and one text session. Deferred replacement also requires cache loan; asynchronous replacement requires pinned expert memory. `STRATA_ORNITH_CACHE_STATIC_COST` freezes measured CPU costs at their initial value for deterministic diagnostics. It is not a recommended performance setting.

`ORNITH_REPLACEMENT` diagnostics report copied bytes, deferred bytes and asynchronous rounds. Slot counts and kernel arithmetic are unchanged by these features.

## Method

Every model process unloads LlamaSwap, checks its running-model list and requires three idle-GPU samples. Tests run serially in a 56 GiB memory scope with swap disabled. No compilation overlaps a timed model run.

Initial screens use greedy 256-token continuations at 4,096/4,097 input tokens. The prose task explains forest recovery after wildfire; the code task requests a Python LRU cache, type annotations and unit tests. They compare budgets of 8, 128 and 512 global replacements per 32 committed tokens, then background replacement with three learned-prior strengths. Instrumented screens select candidates; final confirmation disables routing telemetry and uses separate non-repeating prompt material. Calibration is from the preceding 1,024-token code and retrieval experiment (two repeats of each task), so prose is outside those calibration tasks. The final non-repeating context is separate from the repeated-note calibration context. The LRU programming task itself was also used in calibration: these runs test new context and lengths, not unseen programming tasks. No broad coding-speed claim follows from the LRU result.

The matched Qwen IQ2_XS control retains every routed expert and disables MTP, suffix and lookup drafting. It keeps `--spec 2` as its execution-window setting, supplies no `--mtp` draft module, and records zero drafts offered or accepted. It uses the same input/output lengths and tasks, with its own tokenizer and chat template. Different model outputs are not an identical continuation. Baseline and candidate Ornith runs can also emit different token streams because placement changes numerical paths. These are workload-level throughput comparisons, not fixed-token kernel benchmarks. The separate forced-logit checks assess numerical differences; their startup-inclusive timings are not used as decode-throughput measurements.

Copy equivalence freezes the PCIe quota and CPU-cost ranking, feeds identical tokens, and compares every output logit with synchronous replacement. Another forced comparison measures the numerical change from the actual adaptive candidate versus the starting profile. Neither is a broad model-quality benchmark.

## Why the Qwen control is faster

Local GGUF headers show 48 main layers for both models (Ornith reports 49 including its MTP layer). Qwen has embedding width 2,560, expert hidden width 640 and top-10 routing; Ornith has widths 3,072/1,024 and top-8 routing. Counting the three routed expert projections gives `48 × top_k × 3 × embedding × expert_hidden`: 2.36 billion weight terms per token for Qwen versus 3.62 billion for Ornith, a 1.536× ratio. This excludes attention, shared experts and the output head; it is not total active parameter count or a measured latency ratio. The Qwen control also uses IQ2_XS rather than the Ornith three-bit quants.

The matched Qwen control reserves 18.95 GiB for 14,135 expert slots and reports roughly 89–98% cache hits across these requests. Ornith reserves 11.10 GiB for 3,504 Unsloth slots or 12.27 GiB for 3,216 Bartowski slots. The larger expert blobs and other GPU allocations leave substantially fewer resident experts. The measured Ornith baseline routes roughly 43%/48% of expert entries to the CPU; the strong prior reduces those aggregate fractions to about 36%/40%. Qwen's speed is therefore not evidence that a scheduling switch alone can give Ornith the same throughput.

In a hindsight calculation on the saved code/retrieval routing counts, uniform caches need 156 experts per layer to cover 90% of those accesses: 23.71 GiB of expert storage for Unsloth and 28.58 GiB for Bartowski, before other weights or buffers. This is a fixed-routing calibration estimate, not a minimum-memory proof or a held-out speed prediction; changing placement can also change arithmetic and routes. It nevertheless shows the scale of the residency problem compared with the current 11–12 GiB arenas.

The prior is task-sensitive: its calibration contains code and retrieval. Faster code accompanied by slower prose is not a generally faster configuration. Report the separate tasks and the independent-context confirmation rather than selecting on their pooled average.

## Numerical limits

CPU experts call ggml's weight-specific activation quantizer (`native_quant_act` / `native_quant_h`); the GPU grouped path uses Q8_1 activations. For the K/IQ formats these need not produce the same intermediate values. Moving an expert between CPU and GPU can therefore change logits without changing its stored weights or omitting any routed expert. Timing-dependent placement can add run-to-run variation. In the confirmation repeats, the preserved Unsloth baseline itself emitted different token streams in all four task/input cells, as did the candidate. Consequently, the adaptive forced comparison does not isolate the prior from all existing placement variability; baseline-repeat forced logits and independent held-out text would be needed for a stronger quality assessment. Positional token agreement in free-running text is not a semantic accuracy score. This explains a mechanism, but does not prove that every measured difference is harmless or that quality is unchanged.

Copy equivalence holds placement and scheduling decisions fixed. Adaptive comparisons change placement as well, so they must be reported separately. Baseline-generated forced text also favors the baseline distribution; its NLL is a diagnostic, not an independent accuracy evaluation. The semantic smoke checks are small and do not establish broad quality preservation.

## Remaining memory opportunity

Static analysis found the grouped expert scratch allocated for 4,096 × 8 entries even when large prompts use MMQ. Its size is 420 MiB at FF=1024; 128 hybrid rows require 13.125 MiB. A guarded reduction could recover 406.875 MiB without reducing the prompt chunk. This is not implemented or measured here. It would buy roughly two more resident experts per layer, not enough by itself to explain the Qwen gap.

A separate source/header audit found that input embeddings remain fully GPU-resident: 500.16 MiB of Q5_K for Unsloth and 772.97 MiB of Q8_0 for Bartowski. The output head is a separate 772.97 MiB tensor and should stay on the GPU. Moving only input lookup storage to mapped host memory could free additional cache space, as in the Qwen control, but would add PCIe reads. Graph compatibility, mapping lifetime, fixed-placement parity and prompt throughput need validation before implementation is accepted. This is an unmeasured opportunity, not a speed claim.

A further static audit identified default-on packed Q8 duplicates. The loader keeps both original and packed matrices for eligible shapes; its automatic release only helps when every relevant expert would then fit. Header/eligibility estimates total 1.68 GiB for Unsloth and 0.97 GiB for Bartowski with MTP trimmed. `STRATA_Q8_PACKED=0` is an existing setting to screen before adding allocation code. Dense compute may slow down, so neither these savings nor a speed benefit is claimed as measured. A packed-off plus `STRATA_ORNITH_WIDE=1` arm could then test the existing unpacked-Q8 kernel, with separate numerical validation.

## Results

See [the machine-readable evidence](ornith-residency.json) for exact commands, inputs, outputs, hashes, resource samples, calibration provenance and failed/negative screens.

### Instrumented screens

Throughput cells average the two input lengths from one process and exclude the benchmark warm-up. Routing fractions come from whole-process counters and include that warm-up request. These are selection screens, not final generalization claims.

| Configuration | Prose tokens/s | Code tokens/s | Resident / PCIe / CPU fraction |
| --- | ---: | ---: | --- |
| bartowski-async | 20.93 | 18.59 | 0.321 / 0.205 / 0.474 |
| bartowski-budget128 | 21.01 | 18.94 | 0.415 / 0.192 / 0.393 |
| bartowski-budget512 | 22.16 | 19.24 | 0.443 / 0.162 / 0.396 |
| bartowski-budget8 | 20.88 | 19.16 | 0.321 / 0.198 / 0.481 |
| bartowski-prior16384 | 17.56 | 26.75 | 0.416 / 0.187 / 0.397 |
| bartowski-prior2048 | 18.52 | 22.69 | 0.364 / 0.199 / 0.437 |
| bartowski-prior256 | 18.51 | 18.25 | 0.310 / 0.240 / 0.451 |
| qwen-control | 94.61 | 87.48 | Qwen reports its cache separately |
| unsloth-async | 25.19 | 22.55 | 0.445 / 0.181 / 0.374 |
| unsloth-budget128 | 25.14 | 22.09 | 0.444 / 0.186 / 0.370 |
| unsloth-budget512 | 24.95 | 22.23 | 0.476 / 0.171 / 0.353 |
| unsloth-budget8 | 22.97 | 20.83 | 0.351 / 0.220 / 0.429 |
| unsloth-prior16384 | 20.72 | 31.22 | 0.456 / 0.180 / 0.364 |
| unsloth-prior2048 | 21.52 | 27.74 | 0.437 / 0.188 / 0.375 |
| unsloth-prior256 | 22.85 | 23.10 | 0.447 / 0.194 / 0.359 |

### Confirmation on separate context

Two independent process loads per arm, in opposite order, provide two observations for each task/input-length combination. These runs disable routing telemetry and use non-repeating technical text as context. All outputs must reach 256 tokens; prompt reuse must remain zero. Values are medians; the range is generation throughput across the two loads.

| Quant | Task | Input | Arm | Prompt tokens/s | Generation tokens/s (range) | Response s |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| unsloth | prose | 4096 | baseline | 1231.6 | 21.06 (21.02–21.10) | 15.482 |
| unsloth | prose | 4096 | candidate | 1234.3 | 20.60 (20.50–20.70) | 15.748 |
| unsloth | prose | 4097 | baseline | 1235.0 | 20.55 (20.46–20.65) | 15.776 |
| unsloth | prose | 4097 | candidate | 1217.4 | 20.65 (20.61–20.69) | 15.765 |
| unsloth | code | 4096 | baseline | 1255.7 | 21.23 (21.20–21.27) | 15.321 |
| unsloth | code | 4096 | candidate | 1234.0 | 31.91 (31.48–32.35) | 11.345 |
| unsloth | code | 4097 | baseline | 1235.2 | 21.16 (21.08–21.24) | 15.418 |
| unsloth | code | 4097 | candidate | 1217.5 | 31.07 (31.04–31.10) | 11.607 |
| bartowski | prose | 4096 | baseline | 1029.1 | 18.55 (17.86–19.24) | 17.799 |
| bartowski | prose | 4096 | candidate | 1042.2 | 18.21 (17.55–18.87) | 18.012 |
| bartowski | prose | 4097 | baseline | 1025.8 | 18.47 (17.67–19.27) | 17.884 |
| bartowski | prose | 4097 | candidate | 1029.2 | 18.22 (17.50–18.95) | 18.054 |
| bartowski | code | 4096 | baseline | 1041.8 | 19.06 (18.48–19.65) | 17.377 |
| bartowski | code | 4096 | candidate | 1044.1 | 27.93 (27.05–28.80) | 13.103 |
| bartowski | code | 4097 | baseline | 1025.3 | 19.11 (18.43–19.80) | 17.408 |
| bartowski | code | 4097 | candidate | 1030.5 | 28.48 (28.12–28.83) | 12.969 |

The performance gate requires at least 2% lower total response time and no more than 2% lower prompt throughput in **every** task/input combination. A favorable pooled mean does not override a task regression.

- unsloth: selected `prior16384`; performance gate **False**. Worst prompt ratio 0.9828; worst response ratio 1.0171.
- bartowski: selected `prior16384`; performance gate **False**. Worst prompt ratio 1.0022; worst response ratio 1.0120.

### Correctness and memory

| Quant | Copy equivalence | Default parity | Adaptive forced argmax | Forced NLL baseline → candidate | Mean / max KL |
| --- | --- | --- | --- | --- | --- |
| unsloth | True (128 positions) | True (31,784,960 floats) | 122/128 | 0.076439 → 0.088037 | 0.044240 / 1.240088 |
| bartowski | True (128 positions) | True (31,784,960 floats) | 126/128 | 0.035739 → 0.039110 | 0.008708 / 0.348949 |

Forced NLL uses the baseline-generated continuation. It is not held-out perplexity. Changing residency changes which CPU/GPU arithmetic path executes an expert; byte-identical copy scheduling does not establish byte-identical adaptive generation or unchanged broad model quality.

- unsloth: semantic/length smoke passed `True`; sanitizer/verification passed `True` with 66,816 comparisons, 0 mismatches. Sanitizer uses 32 slots/layer, a 256-token chunk and requests for up to 64 generated tokens after each of 1/8/129/513 prompt tokens to reserve instrumentation memory and exercise replacement boundaries; full runtime allocations are exercised by the normal benchmarks.
- bartowski: semantic/length smoke passed `True`; sanitizer/verification passed `True` with 60,480 comparisons, 0 mismatches. Sanitizer uses 32 slots/layer, a 256-token chunk and requests for up to 64 generated tokens after each of 1/8/129/513 prompt tokens to reserve instrumentation memory and exercise replacement boundaries; full runtime allocations are exercised by the normal benchmarks.

| Quant | Arm | Peak GPU MiB | Peak anonymous + shared GiB | Scope peak GiB | Swap bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| unsloth | baseline | 23794 | 40.23 | 56.00 | 0 |
| unsloth | candidate | 23794 | 40.24 | 56.00 | 0 |
| bartowski | baseline | 23784 | 48.21 | 56.00 | 0 |
| bartowski | candidate | 23784 | 48.21 | 56.00 | 0 |

Scope memory includes reclaimable mapped/file-cache pages; it is not the model’s anonymous working set. CUDA was built and exercised on this GPU. HIP, SYCL, Windows and other GPUs were not built/tested. The CPU tests cover profile rejection/normalization and the existing allocator (six tests total).

### Decision

The general runtime profiles remain unchanged. Neither selected candidate passes the per-task performance gate. The features remain opt-in experiments; they are not a general performance preset.

| LRU-code workload | Baseline generation tokens/s | Candidate generation tokens/s | Response time reduction | Prompt throughput change |
| --- | ---: | ---: | ---: | ---: |
| unsloth | 21.20 | 31.49 | 25.3% | -1.6% |
| bartowski | 19.09 | 28.20 | 25.0% | +0.4% |

These values average the two input-length cell medians above. The calibration-overlapping LRU task improves substantially, but prose has no consistent improvement. Unseen programming tasks were not evaluated. The code gains do not establish unchanged broad model quality: the forced numerical comparisons differ, and both adaptive baselines and candidates vary between process loads. No expert is omitted and model weights are unchanged, but changing the CPU/GPU split changes arithmetic.

First screen the existing packed-Q8-copy tradeoff without the learned prior. The next implementation targets are guarded grouped-scratch compaction, then host-resident input embeddings. Together their audited allocation sizes suggest roughly 0.9–1.2 GiB could become available; neither is implemented or benchmarked in this change. They need fixed-slot parity, automatic-placement quality checks, memory validation and prompt/decode comparisons before promotion.
