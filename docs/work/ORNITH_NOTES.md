# Ornith-1.5-122B: research notes

## Architecture
- Base: `ornith-ai/Ornith-1.5-122B`, HF `Qwen3_5MoeForConditionalGeneration`; GGUF architecture `qwen35moe`
  (the same as Qwen3.6-35B-A3B, which Strata already runs in `src/qwen35`).
- 48 layers + 1 MTP block. Hidden 3072. 3 Gated DeltaNet layers to 1 full-attention layer.
  Attention: 32 heads, 2 KV heads, head 256, partial rope 64, theta 1e7.
  GDN: 16 K heads x 128, 64 V heads x 128, conv 4, state 128.
- MoE: 256 experts, top-8, FF 1024, shared expert FF 1024. About 10B of 124.6B params active per token.

## bartowski IQ3_XS (2 shards, 57.9 GB, sha256 checked)
- Routed experts: IQ3_S (45.8 GiB). blk.47 has gate IQ3_S with up/down Q5_K. MTP blk.48 is Q4_0.
- Dense: Q8_0 / Q6_K (attn_qkv, attn_gate, attn_q), Q8_0 (ssm_out, shexp, output, token_embd).

## What is reused (src/qwen35, strata-q36)
- Loader, GDN/attention kernels, native grouped expert kernels (IQ3_S gate/up), MMQ prompt path, hybrid
  VRAM-slot + pinned-RAM expert cache with CPU pool (ggml-cpu dot products), MTP drafts, server mode.

## Gaps found and fixed
1. IQ3_S was not in the down-projection list (STRATA_D_FMTS); FF 1024 is whole 256-blocks, so it was added.
2. Gate and up have different types in blk.47: the gate is re-encoded to the up's type at load.
3. CPU pool down-activation buffer was 1024 B; Q8_K of 1024 values is 1168 B, so it was raised to 2048.

## Hardware
- RTX 3090 Ti 24 GB, i7-10700 (8C/16T, AVX2, no AVX-512), 61 GB RAM. The 48 GiB of experts does not fit
  in VRAM (about 63 of 256 per layer fit), so decode is bound by the CPU pool and PCIe.
