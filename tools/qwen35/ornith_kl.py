"""tools/qwen35/ornith_kl.py - the engine correctness test for Ornith-1.5-122B (and any qwen35moe GGUF): teacher-forced
full-vocabulary KL of strata-q36 against llama.cpp on the same GGUF.

For each of three prompts, llama.cpp's greedy continuation (64 tokens) is fed to both engines and every step's logits
are compared: KL(llama || strata) over the whole vocabulary, and top-1 agreement.  A second llama.cpp run with another
GPU / CPU split is the noise floor.  Gate: Strata's mean KL <= 1.5x the floor (as tools/qwen35/kl.py), else exit 1.

    python tools/qwen35/ornith_kl.py --model GGUF --strata build-cuda/strata-q36 --dump q36_llama_dump [--out DIR]

q36_llama_dump is tools/qwen35/q36_llama_dump.cpp built against llama.cpp (see its header); --cpu-moe 1 keeps the
experts in RAM so a model larger than VRAM runs.  Measured (RTX 3090 Ti, IQ3_XS): Strata 0.024, floor 0.032.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

V = 248320
PROMPTS = {
    "fr": [760, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 9564, 369],
}
TEXTS = {
    "code": "<|im_start|>user\nWrite a Python function that returns the n-th Fibonacci number iteratively.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "story": "<|im_start|>user\nExplain in three sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
}


def logsm(f: Path) -> np.ndarray:
    x = np.fromfile(f, dtype=np.float32).reshape(-1, V).astype(np.float64)
    x -= x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def kl(p: Path, q: Path) -> tuple[float, float]:
    P, Q = logsm(p), logsm(q)
    n = min(len(P), len(Q))
    k = (np.exp(P[:n]) * (P[:n] - Q[:n])).sum(1)
    return float(k.mean()), float((P[:n].argmax(1) == Q[:n].argmax(1)).mean())


def ids(out: str, key: str = "gen:") -> list[int]:
    line = next(l for l in out.splitlines() if l.startswith(key))
    return [int(x) for x in line[len(key):].split()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--strata", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default="ornith-kl")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"strata": [], "floor": []}
    prompts = dict(PROMPTS)
    for k, t in TEXTS.items():   # llama.cpp's own tokenization (the dump prints the prompt's ids)
        r = subprocess.run([a.dump, "--model", a.model, "--cpu-moe", "1", "--prompt", t, "--n", "1"], capture_output=True, text=True)
        prompts[k] = ids(r.stdout, "prompt:") if "prompt:" in r.stdout else None
        if prompts[k] is None:
            print(f"{k}: could not tokenize with the dump tool", file=sys.stderr)
            return 2
    for k, toks in prompts.items():
        p = ",".join(map(str, toks))
        ref = out / f"ll_{k}.bin"
        r = subprocess.run([a.dump, "--model", a.model, "--cpu-moe", "1", "--tokens", p, "--n", str(a.n),
                            "--dump-full-logits", str(ref)], capture_output=True, text=True, check=True)
        gen = ids(r.stdout)
        force = ",".join(map(str, gen[:-1]))
        flo = out / f"ll2_{k}.bin"
        subprocess.run([a.dump, "--model", a.model, "--ngl", "14", "--tokens", p, "--force", force, "--n", str(a.n),
                        "--dump-full-logits", str(flo)], capture_output=True, text=True, check=True)
        st = out / f"st_{k}.bin"
        subprocess.run([a.strata, "--model", a.model, "--max-context", "4096", "--tokens", p, "--force", force,
                        "--dump-full-logits", str(st)], capture_output=True, text=True, check=True)
        ks, ts = kl(ref, st)
        kf, tf = kl(ref, flo)
        res["strata"].append(ks)
        res["floor"].append(kf)
        print(f"{k}: strata KL {ks:.4f} top1 {100 * ts:.1f}% | floor KL {kf:.4f} top1 {100 * tf:.1f}%")
    s, f = float(np.mean(res["strata"])), float(np.mean(res["floor"]))
    ok = s <= 1.5 * f
    print(f"mean KL: strata {s:.4f}, floor {f:.4f} ({s / f:.2f}x): {'PASS' if ok else 'FAIL'} (gate 1.5x)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
