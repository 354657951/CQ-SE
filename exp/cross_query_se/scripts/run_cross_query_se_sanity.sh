#!/bin/bash
# Sanity check: cross-query SE on NQ only, 50 examples, 1 seed, 1 GPU for vLLM.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a
echo "=== Cross-Query SE Sanity Check ==="
nvidia-smi | head -5

python cross_query_se/scripts/run_cross_query_se.py \
  --datasets nq \
  --output_dir cross_query_se/outputs/cross_query_se_sanity \
  --results_dir cross_query_se/results/cross_query_se_sanity \
  --k_perturb 10 \
  --seeds 0 \
  --dev_size 30 \
  --num_samples 50 \
  --top_k 5 \
  --top_k_dual 5 \
  --chunk_size 500000 \
  --vllm_tp 1 \
  --gpu_id 0

echo "=== Sanity Check Done ==="
cat cross_query_se/results/cross_query_se_sanity/cross_query_se_results.json
