# Ornith packed-Q8 memory tradeoff

Setting `STRATA_Q8_PACKED=0` made the tested Ornith workloads faster on an RTX 3090 Ti by freeing duplicate packed dense weights for a larger expert cache. This is an existing setting; no engine code or production serving profile was changed. The gain does not establish broad accuracy preservation or performance on other tasks.

## Measurement

Tested engine revision `f785a245377f1fd569b947e9e59d9a4d52a8d1b4`, CUDA 13, RTX 3090 Ti 24 GiB, i7-10700, 64 GB installed RAM, PCIe 3 x16. Both arms used the current optimized profile, MTP off, context 8192, chunk 4096, seven pool workers, adaptive PCIe misses, and cache loan. No learned prior, asynchronous cache replacement, deferred replacement or wide kernel option was added. Only `STRATA_Q8_PACKED` differed: explicitly 1 versus 0.

Two independent loads per arm, reversing within-model order in the second pass. Each load had an excluded warm-up followed by prose and interval-merging code requests at 4096 and 4097 input tokens, each generating 256 tokens. All 32 measured requests had zero prompt reuse and zero drafts. The coding task was not used to train the previous cache prior. Context material and tokenized inputs were identical between arms. The exact task, harness, commands, inputs, outputs, logs and resource samples are in [the evidence](ornith-packed-q8.json).

The table averages four per-task/length medians across the two loads. With only two observations per cell, these are descriptive measurements, not confidence intervals.

| Quant | Generation tokens/s, on → off | Prompt tokens/s, on → off | Response seconds, on → off |
|---|---:|---:|---:|
| Unsloth | 20.57 → 21.82 | 1239.3 → 1234.8 | 15.757 → 15.059 |
| Bartowski | 17.94 → 19.82 | 1025.3 → 1029.1 | 18.272 → 16.902 |

Unsloth's per-cell response time fell 3.84–4.97%; Bartowski's fell 7.25–7.76%. The largest median prompt-throughput decrease was 1.36% (Unsloth prose at 4096); Bartowski's largest decrease was 0.12%. Both pass the exploratory performance gate of at least 2% lower response time and prompt throughput at least 98% of baseline in every tested cell. This tolerance does not mean zero prompt cost.

## Memory

Unsloth's expert arena increased from 11.10 to 12.77 GiB, 73 to 84 experts per layer. Bartowski's increased from 12.27 to 13.19 GiB, 67 to 72 experts per layer. The saved dense-weight memory is reused for experts, so this setting does not substantially reduce total memory use.

Completed timing runs used approximately 40.24 GiB (Unsloth) and 48.21 GiB (Bartowski) peak process anonymous plus shared host memory. GPU peaks were 23794 → 23744 MiB and 23784 → 23690 MiB respectively. All runs used a 56 GiB cgroup limit with swap disabled and recorded zero swap. File-backed cache is additional to anonymous/shared memory; the cgroup limit includes it. These measurements are not total machine RAM use.

## Validation and limits

Passed: all eight model loads and 32 measured requests, matching inputs, 256 outputs per request, uncached requests, no speculative drafts, zero scope swap. Before each model process, LlamaSwap was unloaded and three GPU samples showed below 100 MiB and zero utilization.

Timing outputs are coherent but code is cut off at the 256-token limit; they are not completed coding tests. Free-running token streams can change with kernel and expert placement. No full-logit parity, held-out perplexity, broad accuracy benchmark, packed-off plus wide-kernel test, or compute-sanitizer run was performed in this round. Earlier packed-on validation does not validate the changed dispatch. No new build was needed. HIP, SYCL, Windows, MTP and concurrent serving were not tested.

Failed setup attempt: the first launcher omitted the tokenizer module directory and exited before model startup. The path was corrected; no failed model request is hidden.

## Next work

Keep packed-off as an experimental candidate until matched-placement numerical checks and broader quality tests are complete. Test packed-off plus the existing wide-Q8 option as a separate arm. Then implement guarded grouped-scratch compaction and evaluate host-resident input embeddings, each with independent controls. Keep the output head on the GPU. No 70-token/s claim follows from this experiment.

Packed-off semantic smoke checks passed for both quants: arithmetic 17×23=391, a Python square-function AST check, retrieval of ORNITH-724, and output limits of 1/2/3 tokens. The sky-explanation outputs are retained for inspection. These checks are narrow and do not establish broad quality parity.
