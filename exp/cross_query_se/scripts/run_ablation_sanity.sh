#!/bin/bash
# Smoke-test for all three ablation scripts: NQ only, 5 examples, seed 0.
# Writes to ablation_*_sanity/ dirs to avoid polluting real output dirs.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

echo "=== Ablation Sanity Check ==="
nvidia-smi | head -5

BASE_DIR="cross_query_se/outputs/cross_query_se"

echo "--- Sanity 1: w/o re-retrieval ---"
# Skip stage 8 (enhanced answer generation requires full corpus scan, too slow for sanity)
VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_ablation_no_reretrieval.py \
  --datasets nq \
  --output_dir cross_query_se/outputs/ablation_no_reretrieval_sanity \
  --base_output_dir "$BASE_DIR" \
  --results_dir cross_query_se/results/ablation_no_reretrieval_sanity \
  --seeds 0 \
  --dev_size 2 \
  --max_examples 5 \
  --vllm_tp 4 \
  --gpu_id 4 \
  --chunk_size 500000 \
  --stages 3 4 5 51 7 9

echo "--- Sanity 2: paraphrase-only ---"
# Skip stage 8 for the same reason
VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_ablation_paraphrase_only.py \
  --datasets nq \
  --output_dir cross_query_se/outputs/ablation_paraphrase_only_sanity \
  --base_output_dir "$BASE_DIR" \
  --results_dir cross_query_se/results/ablation_paraphrase_only_sanity \
  --seeds 0 \
  --dev_size 2 \
  --max_examples 5 \
  --vllm_tp 4 \
  --gpu_id 0 \
  --chunk_size 500000 \
  --stages 2 3 4 5 51 7 9

echo "--- Sanity 3: tau sweep (tau=0.85) ---"
VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_ablation_tau_sweep.py \
  --datasets nq \
  --tau 0.85 \
  --base_output_dir "$BASE_DIR" \
  --results_dir cross_query_se/results/ablation_tau_sweep_sanity \
  --seeds 0 \
  --dev_size 2 \
  --max_examples 5 \
  --vllm_tp 4 \
  --gpu_id 0 \
  --chunk_size 500000 \
  --stages 2 3 4 5 51 7 9

echo "=== All sanity checks passed ==="
