#!/bin/bash
# INTRYGUE-style induction-aware entropy baseline on Qwen2.5-72B-Instruct: all 5 datasets, 3 seeds, 8 GPUs.
# Stages 1-2: HF eager attention with device_map=auto (72B model spans all 8 GPUs).
# Stage 5: vLLM enhanced generation with tensor_parallel_size=4.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a
echo "=== INTRYGUE Baseline 72B Full Run ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5

python cross_query_se/scripts/run_intrygue_baseline.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --datasets nq webqa triviaqa hotpotqa squad \
  --output_dir cross_query_se/outputs/intrygue_baseline_72b \
  --results_dir cross_query_se/results/intrygue_baseline_72b \
  --sugar_output_dir cross_query_se/outputs/sugar_baseline_72b \
  --seeds 0 1 2 \
  --dev_size 500 \
  --top_k 5 \
  --top_k_heads 10 \
  --n_calib 50 \
  --vllm_tp 4 \
  --stages 5 6

echo "=== Full Run Done ==="
ls -lh cross_query_se/outputs/intrygue_baseline_72b/
ls -lh cross_query_se/results/intrygue_baseline_72b/
