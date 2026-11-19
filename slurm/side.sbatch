#!/bin/bash
# Q4 side-effect matrix: each trait vector and three random directions on neutral
# prompts, below (LOW) and above (HIGH) the coherence break; all ten rubrics.
#   sbatch --export=ALL,MODEL=qwen3-4b,LOW=0.3,HIGH=1.5 slurm/side.sbatch
#SBATCH --job-name=side
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
export HF_HUB_OFFLINE=1   # weights come from slurm/download.sbatch
MODEL=${MODEL:?}; LOW=${LOW:?}; HIGH=${HIGH:?}; JUDGE=${JUDGE:-mistral-small}
for VSET in A B C; do
  for T in "${TRAITS[@]}"; do
    $PY -m steer.generate --model $MODEL --trait $T --vset $VSET --grid main --tag side --prompts neutral \
        --n-prompts 40 --strengths 0 $LOW $HIGH --no-greedy --batch-size 128
  done
done
for R in random0 random1 random2; do
  $PY -m steer.generate --model $MODEL --trait $R --vset A --grid main --tag side --prompts neutral \
      --n-prompts 40 --strengths 0 $LOW $HIGH --no-greedy --batch-size 128
done
$PY -m steer.judge --judge $JUDGE --metrics all --batch-size 32 --input "$STEER_OUT"/gen/side/$MODEL/*.jsonl
$PY -m analysis.q4_scale side --model $MODEL --low $LOW --high $HIGH
