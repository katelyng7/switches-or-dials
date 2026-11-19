# Switches or Dials?

Code for "Switches or Dials? How Language Model Traits Respond to Activation
Steering" (Team ULP, MIT NLP, Fall 2026). Plan: `docs/plan.pdf`. Paper draft:
`main.tex`.

Nothing here trains a model. Steering vectors are differences of averaged
activations; everything else is generation and scoring.

## Layout

| Path | What it does |
|---|---|
| `configs/` | traits and their groups, model names, strength grids |
| `data/traits/<trait>/` | three extraction sets (A, B, C) and 120 held-out prompts |
| `steer/` | model loading, steering hook, extraction, generation, judge |
| `analysis/` | Q1 fit, greedy check, Q2 test, Q3 curves, Q4 scale and side effects |
| `slurm/` | job scripts for MIT Engaging |
| `scripts/smoke.sh` | laptop end-to-end check with tiny models |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q          # hook and fit tests, about 10 s
.venv/bin/python -m steer.validate_data
./scripts/smoke.sh                     # downloads Qwen3-0.6B and 1.7B
```

## Running on Engaging

1. Clone the repo, create the venv, and edit the paths in `slurm/env.sh`.
2. `huggingface-cli login` with a token that has accepted the Mistral-Small
   and Gemma 3 licences.
3. Pilot (sets the strength grid):
   `sbatch -p <partition> -A <account> --array=0,5,7 --export=ALL,MODEL=qwen3-4b,GRID=pilot slurm/pipeline.sbatch`
4. Replace `grids.main.strengths` in `configs/grid.yaml` using the pilot, then
   the main grid, once per vector set:
   `sbatch ... --array=0-9 --export=ALL,MODEL=qwen3-4b,VSET=A,GRID=main slurm/pipeline.sbatch`
5. Analysis (CPU is enough):

```bash
python -m steer.extract agree --model qwen3-4b        # vector agreement A/B/C
python -m analysis.hierfit --model qwen3-4b           # Q1: mu, sigma_t, w, S
python -m analysis.greedy --model qwen3-4b            # greedy check on sigma_t
sbatch ... slurm/second_judge.sbatch                  # judge agreement
python -m analysis.q2_groups --model qwen3-4b         # Q2
sbatch ... slurm/q3.sbatch; python -m analysis.q3_commit --model qwen3-4b
sbatch ... slurm/side.sbatch                          # Q4 side effects
python -m analysis.q4_scale ladder --models qwen3-0.6b qwen3-1.7b qwen3-4b qwen3-8b qwen3-14b
```

Every stage appends to its output file and skips rows already present, so a
job that hits the time limit is resumed by submitting it again.

## Conventions

- Steered layer: `num_layers // 2`. Strength `a` multiplies the median
  residual norm at that layer; the vector is unit-normalised.
- The last prompt token counts as generation position 1 (see `steer/hooks.py`).
- A reply expresses a trait, or is coherent, when the judge's expected rating
  is at least 5 on the 0-9 scale.

## Credits

`data/upstream/` and the `extract_A.json` files for hallucinating,
sycophantic, humorous, optimistic, and impolite come from
[safety-research/persona_vectors](https://github.com/safety-research/persona_vectors)
(Apache-2.0). The extraction recipe follows Chen et al. (2025).
