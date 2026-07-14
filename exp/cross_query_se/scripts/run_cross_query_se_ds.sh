#!/bin/bash
# Cross-query SE for a specific dataset and specific stages.
# Usage: run_cross_query_se_ds.sh <dataset> <start_stage> <end_stage>
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

DATASET=${1:-nq}
START_STAGE=${2:-1}
END_STAGE=${3:-9}

echo "=== Cross-Query SE: dataset=$DATASET stages=$START_STAGE-$END_STAGE ==="
nvidia-smi | head -5

# Build stages list
STAGES=$(seq $START_STAGE $END_STAGE | tr '\n' ' ')
echo "Stages: $STAGES"

python cross_query_se/scripts/run_cross_query_se.py \
  --datasets $DATASET \
  --output_dir cross_query_se/outputs/cross_query_se \
  --results_dir cross_query_se/results/cross_query_se \
  --k_perturb 10 \
  --seeds 0 1 2 \
  --dev_size 500 \
  --top_k 5 \
  --top_k_dual 5 \
  --chunk_size 500000 \
  --vllm_tp 4 \
  --gpu_id 4 \
  --query_batch_size 800 \
  --stages $STAGES

echo "=== Done: dataset=$DATASET stages=$START_STAGE-$END_STAGE ==="
