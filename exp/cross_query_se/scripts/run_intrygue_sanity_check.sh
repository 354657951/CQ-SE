#!/bin/bash
# Sanity check: INTRYGUE baseline on nq only, 60 examples, stages 1-6.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

echo "=== INTRYGUE Sanity Check ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5

python cross_query_se/scripts/run_intrygue_baseline.py \
  --datasets nq \
  --output_dir cross_query_se/outputs/intrygue_sanity \
  --results_dir cross_query_se/results/intrygue_sanity \
  --sugar_output_dir cross_query_se/outputs/sugar_baseline \
  --seeds 0 \
  --dev_size 50 \
  --top_k 5 \
  --hf_gpu_id 7 \
  --vllm_tp 4 \
  --top_k_heads 10 \
  --n_calib 20 \
  --num_samples 60 \
  --stages 1 2 3 4 5 6

echo "=== Sanity Check Done ==="
ls -lh cross_query_se/outputs/intrygue_sanity/
ls -lh cross_query_se/results/intrygue_sanity/
