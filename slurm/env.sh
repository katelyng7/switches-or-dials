# Sourced by every job. Defaults follow the MIT ORCD layout (pool storage under
# ~/orcd/pool); override any of them in your shell before sbatch if needed.
export PROJECT_DIR=${PROJECT_DIR:-$HOME/orcd/pool/switches-or-dials}
export HF_HOME=${HF_HOME:-$HOME/orcd/pool/hf_cache}     # model weights (large)
export STEER_OUT=${STEER_OUT:-$PROJECT_DIR/out}         # all outputs
export PY=${PY:-$PROJECT_DIR/.venv/bin/python}
export TOKENIZERS_PARALLELISM=false
TRAITS=(refusing sycophantic hallucinating deceptive clarifying humorous optimistic poetic neurotic impolite)
cd "$PROJECT_DIR"
