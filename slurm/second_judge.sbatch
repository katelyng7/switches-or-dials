#!/bin/bash
# Qwen3-14B re-scores a fixed 10% of the main grid for one steered model.
#   sbatch --export=ALL,MODEL=qwen3-4b slurm/second_judge.sbatch
#SBATCH --job-name=judge2
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
export HF_HUB_OFFLINE=1   # weights come from slurm/download.sbatch
MODEL=${MODEL:?}; GRID=${GRID:-main}
$PY -m steer.judge --judge qwen3-14b --subset 0.1 --batch-size 32 --input "$STEER_OUT"/gen/$GRID/$MODEL/*.jsonl
$PY -m analysis.judge_agreement --model $MODEL --tag $GRID
