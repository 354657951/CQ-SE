#!/bin/bash
# Sanity check: NQ only, 550 examples, seed=0.
# Verifies all 7 stages run correctly before full experiment.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a
echo "=== SUGAR Baseline Sanity Check ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5

python cross_query_se/scripts/run_sugar_baseline.py \
  --datasets nq \
  --output_dir cross_query_se/outputs/sugar_baseline_sanity \
  --results_dir cross_query_se/results/sugar_baseline_sanity \
  --m_samples 5 \
  --seeds 0 \
  --dev_size 500 \
  --top_k 5 \
  --top_k_dual 5 \
  --chunk_size 500000 \
  --vllm_tp 4 \
  --gpu_id 4 \
  --num_samples 550

echo "=== Sanity Check Done ==="
ls -lh cross_query_se/outputs/sugar_baseline_sanity/
ls -lh cross_query_se/results/sugar_baseline_sanity/
