#!/usr/bin/env python3
"""Prepare the fork's opt-in, text-only Ornith server config. No engine download."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

MODELS = {
    "unsloth-iq3-xxs": {
        "repo": "unsloth/Ornith-1.5-122B-GGUF",
        "revision": "3842d003341507c216295c6f28dfcd30349925c6",
        "files": [("Ornith-1.5-122B-UD-IQ3_XXS.gguf", 49515667712,
                   "027a9a480e8bcc6079fcf834e368467a169953561fb4b98dc9528d6185c8a438")],
    },
    "bartowski-q3-k-s": {
        "repo": "bartowski/Ornith-1.5-122B-GGUF",
        "revision": "a8bc08fa33fc3a3e79380bf661a84688c7ea0a50",
        "files": [
            ("Ornith-1.5-122B-Q3_K_S/Ornith-1.5-122B-Q3_K_S-00001-of-00002.gguf", 39879884352,
             "7fa59f1e11bd41c9a8afdf971411203d5a51208c43ef14bb6967fce796779d3a"),
            ("Ornith-1.5-122B-Q3_K_S/Ornith-1.5-122B-Q3_K_S-00002-of-00002.gguf", 17600078784,
             "7b163ce201c23d975862acefb89995af1068cce5f401a80fe34e93f1892abf58"),
        ],
    },
}

def verified(path: Path, size: int, digest: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == digest

def prepare(args):
    model = MODELS[args.quant]
    engine = args.engine.resolve()
    if not engine.is_file():
        raise SystemExit(f"Build this fork's CUDA engine first: {engine}")
    folder = args.models.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for remote, size, digest in model["files"]:
        path = folder / Path(remote).name
        print(f"Checking {path.name} ({size / 1e9:.2f} GB)", flush=True)
        if not verified(path, size, digest):
            if path.exists():
                raise SystemExit(f"Size or SHA-256 mismatch: {path}. Existing files are never overwritten.")
            if not args.download:
                raise SystemExit(f"Missing {path}; supply --download to fetch the pinned files.")
            part = path.with_suffix(path.suffix + ".part")
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{remote}"
            print(f"Downloading {remote} ({size / 1e9:.2f} GB)", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
            if not verified(part, size, digest):
                raise SystemExit(f"Download failed integrity verification: {part}")
            part.rename(path)
        files.append(path)
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(root / "tools/strata_tokenizer.py"),
                    "--gguf", str(files[0]), "--out", str(output), "--check"], check=True)
    # Extract the embedded Ornith chat template with its tokenizer; never substitute Qwen's defaults.
    config = {"exe": str(engine), "cwd": str(root), "arch": "qwen35moe",
              "args": ["--native", str(files[0]), "--max-context", "8192", "--chunk", "32",
                       "--mtp", "0", "--pool-workers", "7"],
              "env": {"STRATA_ORNITH": "1"}, "tokenizer": str(output / "tokenizer"),
              "model_name": "ornith-1.5-122b-" + args.quant, "host": "127.0.0.1",
              "port": args.port, "open_browser": False, "sampling": {"temperature": 0.0}}
    if args.lib_dir:
        config["lib_dirs"] = [str(path.resolve()) for path in args.lib_dir]
    config_path = output / "server.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Prepared {config_path}\nStart: {sys.executable} {root / 'serve/server.py'} --engine strata --config {config_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant", choices=MODELS, required=True)
    parser.add_argument("--models", type=Path, required=True, help="directory containing both split files, if applicable")
    parser.add_argument("--engine", type=Path, required=True, help="CUDA strata binary compiled from this fork")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--lib-dir", type=Path, action="append", default=[],
                        help="extra engine/CUDA library directory; may be repeated")
    parser.add_argument("--port", type=int, default=8080)
    prepare(parser.parse_args())

if __name__ == "__main__":
    main()
