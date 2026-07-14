#!/bin/bash
# SUGAR-style within-query SE baseline on Qwen2.5-72B-Instruct: all 5 datasets, 3 seeds, 8 GPUs.
# Uses tensor_parallel_size=4 (GPUs 4-7 for vLLM 72B; GPUs 0-3 for BGE/DeBERTa).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a
echo "=== SUGAR Baseline 72B Full Run ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5

python cross_query_se/scripts/run_sugar_baseline.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --datasets nq webqa triviaqa hotpotqa squad \
  --output_dir cross_query_se/outputs/sugar_baseline_72b \
  --results_dir cross_query_se/results/sugar_baseline_72b \
  --m_samples 5 \
  --seeds 0 1 2 \
  --dev_size 500 \
  --top_k 5 \
  --top_k_dual 5 \
  --chunk_size 500000 \
  --vllm_tp 4 \
  --gpu_id 4

echo "=== Full Run Done ==="
ls -lh cross_query_se/outputs/sugar_baseline_72b/
ls -lh cross_query_se/results/sugar_baseline_72b/
