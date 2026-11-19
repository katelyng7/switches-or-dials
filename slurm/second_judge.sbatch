#!/bin/bash
# Qwen3-14B re-scores a fixed 10% of the main grid for one steered model.
#   sbatch -p <partition> -A <account> --export=ALL,MODEL=qwen3-4b slurm/second_judge.sbatch
#SBATCH --job-name=judge2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
MODEL=${MODEL:?}; GRID=${GRID:-main}
$PY -m steer.judge --judge qwen3-14b --subset 0.1 --batch-size 32 --input "$STEER_OUT"/gen/$GRID/$MODEL/*.jsonl
$PY -m analysis.judge_agreement --model $MODEL --tag $GRID
