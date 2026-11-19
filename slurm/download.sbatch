#!/bin/bash
# Download model weights on a CPU node so GPU jobs can run offline.
#   sbatch --export=ALL,MODELS="qwen3-4b mistral-small" slurm/download.sbatch
#SBATCH --job-name=dl
#SBATCH -p mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
$PY - $MODELS <<'PY'
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from steer.common import load_yaml
table = load_yaml("models.yaml")
ids = {**{k: v["hf"] for k, v in table["steered"].items()}, **{k: v["hf"] for k, v in table["judges"].items()}}
for short in sys.argv[1:]:
    hf = ids[short]
    path = snapshot_download(hf, ignore_patterns=["consolidated*", "*.gguf", "original/*"])
    tok = AutoTokenizer.from_pretrained(hf)
    digits = [tok.encode(d, add_special_tokens=False) for d in "0123456789"]
    print(short, hf, "->", path, "| digits single-token:", all(len(d) == 1 for d in digits), flush=True)
PY
$PY -m pytest -q tests
