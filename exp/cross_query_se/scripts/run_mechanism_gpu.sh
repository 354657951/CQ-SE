#!/bin/bash
# Full run: token-NLL (all datasets) + random perturbation sanity check (1000 examples/dataset).
# Uses 4 GPUs (vllm_tp=4 for both token-NLL and random-pert vLLM stages).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

echo "=== Mechanism GPU Job ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5

echo "--- Step 1: Token-NLL (all datasets, full test set, vllm_tp=4) ---"
# Skips datasets where output already exists (checkpointed)
VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_token_nll.py \
    --datasets nq webqa triviaqa hotpotqa squad \
    --base_dir cross_query_se/outputs/cross_query_se \
    --output_dir cross_query_se/outputs/token_nll \
    --dev_size 500 \
    --num_samples 999999 \
    --batch_size 512 \
    --vllm_tp 4

echo "--- Step 2: Random Perturbation Sanity Check (all datasets, 1000 examples, vllm_tp=4) ---"
# chunk_size=10M to reduce number of corpus scan passes (fewer I/O operations = faster)
VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_random_pert_sanity.py \
    --datasets nq webqa triviaqa hotpotqa squad \
    --output_dir cross_query_se/outputs/random_pert_sanity \
    --results_dir cross_query_se/results/random_pert_sanity \
    --seeds 0 1 2 \
    --dev_size 500 \
    --num_samples 1000 \
    --top_k 5 \
    --vllm_tp 4 \
    --gpu_id 0 \
    --chunk_size 3000000

echo "=== Mechanism GPU Job Done ==="
