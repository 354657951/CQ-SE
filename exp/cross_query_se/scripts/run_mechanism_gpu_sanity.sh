#!/bin/bash
# Sanity check: token-NLL + random perturbation on NQ only, 50 examples, 1 GPU.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

echo "=== Mechanism GPU Sanity Check ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5

echo "--- Step 1: Token-NLL (NQ, 50 examples, 1 GPU) ---"
VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_token_nll.py \
    --datasets nq \
    --base_dir cross_query_se/outputs/cross_query_se \
    --output_dir cross_query_se/outputs/token_nll_sanity \
    --dev_size 500 \
    --num_samples 50 \
    --batch_size 50 \
    --vllm_tp 1

echo "--- Step 2: Random Perturbation Sanity (NQ, 50 examples, 1 GPU) ---"
VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_random_pert_sanity.py \
    --datasets nq \
    --output_dir cross_query_se/outputs/random_pert_sanity_debug \
    --results_dir cross_query_se/results/random_pert_sanity_debug \
    --seeds 0 \
    --dev_size 500 \
    --num_samples 50 \
    --top_k 5 \
    --vllm_tp 1 \
    --gpu_id 0 \
    --chunk_size 500000

echo "=== Sanity Check Done ==="
