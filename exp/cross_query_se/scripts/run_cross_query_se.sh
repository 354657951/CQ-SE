#!/bin/bash
# Full cross-query SE pipeline: all 5 datasets, 3 seeds, 8 GPUs.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a
echo "=== Cross-Query SE Full Run ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5

python cross_query_se/scripts/run_cross_query_se.py \
  --datasets nq webqa triviaqa hotpotqa squad \
  --output_dir cross_query_se/outputs/cross_query_se \
  --results_dir cross_query_se/results/cross_query_se \
  --k_perturb 10 \
  --seeds 0 1 2 \
  --dev_size 500 \
  --top_k 5 \
  --top_k_dual 5 \
  --chunk_size 500000 \
  --vllm_tp 4 \
  --gpu_id 4

echo "=== Full Run Done ==="
ls -lh cross_query_se/outputs/cross_query_se/
ls -lh cross_query_se/results/cross_query_se/
