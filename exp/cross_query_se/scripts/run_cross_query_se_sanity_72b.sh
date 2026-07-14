#!/bin/bash
# Sanity check: cross-query SE on Qwen2.5-72B-Instruct, NQ only, 50 examples, 1 seed, 4 GPUs (TP=4).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a
echo "=== Cross-Query SE Sanity Check 72B ==="
nvidia-smi | head -5

VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_cross_query_se.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --datasets nq \
  --output_dir cross_query_se/outputs/cross_query_se_sanity_72b \
  --results_dir cross_query_se/results/cross_query_se_sanity_72b \
  --k_perturb 10 \
  --seeds 0 \
  --dev_size 30 \
  --num_samples 50 \
  --top_k 5 \
  --top_k_dual 5 \
  --chunk_size 500000 \
  --vllm_tp 4 \
  --gpu_id 4

echo "=== Sanity Check 72B Done ==="
cat cross_query_se/results/cross_query_se_sanity_72b/cross_query_se_results.json
