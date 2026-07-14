#!/bin/bash
# Full INTRYGUE-style induction-aware entropy baseline: all 5 datasets, 3 seeds, 8 GPUs.
# GPU 7 for HF eager-attention model (stages 1-2); GPUs 0-3 for vLLM (stage 5).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

echo "=== INTRYGUE Baseline Full Run ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5

python cross_query_se/scripts/run_intrygue_baseline.py \
  --datasets nq webqa triviaqa hotpotqa squad \
  --output_dir cross_query_se/outputs/intrygue_baseline \
  --results_dir cross_query_se/results/intrygue_baseline \
  --sugar_output_dir cross_query_se/outputs/sugar_baseline \
  --seeds 0 1 2 \
  --dev_size 500 \
  --top_k 5 \
  --hf_gpu_id 7 \
  --vllm_tp 4 \
  --top_k_heads 10 \
  --n_calib 50 \
  --stages 1 2 3 4 5 6

echo "=== Full Run Done ==="
ls -lh cross_query_se/outputs/intrygue_baseline/
ls -lh cross_query_se/results/intrygue_baseline/
