#!/bin/bash
# End-to-end check on a laptop with Qwen3-0.6B as both steered model and judge.
# The numbers mean nothing; this only proves every stage runs and resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
export STEER_OUT="$PWD/out/smoke"
PY=${PY:-.venv/bin/python}
M=qwen3-0.6b; T=${1:-humorous}; J=smoke
$PY -m steer.extract gen --model $M --trait $T --vset A --n-questions 6 --reps 1 --max-new-tokens 60 --batch-size 16
$PY -m steer.judge --judge $J --input "$STEER_OUT/extract/$M/${T}_A.jsonl"
$PY -m steer.extract vec --model $M --trait $T --vset A --judge $J --min-pairs 2
$PY -m steer.generate --model $M --trait $T --vset A --grid smoke --batch-size 16
$PY -m steer.judge --judge $J --input "$STEER_OUT/gen/smoke/$M/${T}_A.jsonl"
$PY -m analysis.hierfit --model $M --tag smoke --judge $J --boot 20
