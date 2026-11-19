# Sourced by every job. Edit the three paths for your Engaging account.
export PROJECT_DIR=${PROJECT_DIR:-$HOME/switches-or-dials}
export HF_HOME=${HF_HOME:-/path/to/scratch/$USER/hf}          # model weights (large)
export STEER_OUT=${STEER_OUT:-/path/to/scratch/$USER/steer_out} # all outputs
export PY=${PY:-$PROJECT_DIR/.venv/bin/python}
TRAITS=(refusing sycophantic hallucinating deceptive clarifying humorous optimistic poetic neurotic impolite)
cd "$PROJECT_DIR"
