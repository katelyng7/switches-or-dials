#!/bin/bash
# One trait, one vector set, one model: extract -> judge -> vector -> grid -> judge.
# Every stage resumes, so if the job hits the time limit just submit it again.
#
#   sbatch --array=0-9 \
#          --export=ALL,MODEL=qwen3-4b,VSET=A,GRID=main slurm/pipeline.sbatch
#
#SBATCH --job-name=steer
#SBATCH -p mit_preemptable
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
source slurm/env.sh
export HF_HUB_OFFLINE=1   # weights come from slurm/download.sbatch
MODEL=${MODEL:?}; VSET=${VSET:-A}; GRID=${GRID:-main}; JUDGE=${JUDGE:-mistral-small}
T=${TRAITS[$SLURM_ARRAY_TASK_ID]}

$PY -m steer.extract gen --model $MODEL --trait $T --vset $VSET --batch-size 128
$PY -m steer.judge --judge $JUDGE --input "$STEER_OUT/extract/$MODEL/${T}_${VSET}.jsonl" --batch-size 32
$PY -m steer.extract vec --model $MODEL --trait $T --vset $VSET --judge $JUDGE
$PY -m steer.generate --model $MODEL --trait $T --vset $VSET --grid $GRID --batch-size 128
$PY -m steer.judge --judge $JUDGE --input "$STEER_OUT/gen/$GRID/$MODEL/${T}_${VSET}.jsonl" --batch-size 32
