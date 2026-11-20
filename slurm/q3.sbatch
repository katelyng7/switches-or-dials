#!/bin/bash
# Q3 commitment runs for one trait at one strength (set A to roughly exp(mu) from Q1).
#   sbatch --export=ALL,MODEL=qwen3-4b,TRAIT=refusing,A=0.4 slurm/q3.sbatch
#SBATCH --job-name=q3
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
export HF_HUB_OFFLINE=1   # weights come from slurm/download.sbatch
MODEL=${MODEL:?}; TRAIT=${TRAIT:?}; A=${A:?}; VSET=${VSET:-A}; JUDGE=${JUDGE:-mistral-small}
G="$PY -m steer.generate --model $MODEL --trait $TRAIT --vset $VSET --grid main --tag q3 --strengths 0 $A --no-greedy --batch-size 128"
$G --mode response            # reference: steer every generation position
$G --mode prompt
for K in 1 2 4 8 16 32; do
  $G --mode first_k --k $K
  $G --mode after_k --k $K
done
$PY -m steer.judge --judge $JUDGE --input "$STEER_OUT/gen/q3/$MODEL/${TRAIT}_${VSET}.jsonl" --batch-size 32
