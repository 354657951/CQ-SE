#!/bin/bash
# Optimization run: fix three bugs in cross-query SE pipeline.
# Bug 1: AUROC sign inversion (was roc_auc_score(correct, H_cq); fixed to use -H_cq)
# Bug 2: Answer selection was best_by_relevance (mostly picks orig query); fixed to majority vote
# Bug 3: Threshold tuning used enhanced_answer=None -> collapsed to rag3; fixed to use majority_answer
#
# Strategy: reuse cached stage 1-6 outputs; only re-run stages 5b, 7, 8, 9.
# Run stages 5b+7 first (no GPU needed for 5b; stage 7 uses GPU only for select_topk_of_query_info).
# Then run stage 8+9 in a FRESH subprocess to avoid CUDA fork issue with vLLM.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

OUTPUT_DIR="cross_query_se/outputs/cross_query_se"
RESULTS_DIR="cross_query_se/results/cross_query_se_opt"

echo "=== Cross-Query SE Optimization: Stages 5b, 7, 8, 9 ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5

# Stage 5b and 7 (thresholds + AIS - stages 5b and 7 are already done, will skip)
echo "--- Running stages 5b, 7 (CPU-only threshold tuning + AIS, may skip if already done) ---"
python cross_query_se/scripts/run_cross_query_se.py \
    --datasets nq webqa triviaqa hotpotqa squad \
    --output_dir "${OUTPUT_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --seeds 0 1 2 \
    --dev_size 500 \
    --top_k 5 \
    --top_k_dual 5 \
    --vllm_tp 4 \
    --gpu_id 0 \
    --stages 51 7

echo "--- Stages 5b, 7 complete ---"

# Stage 8 (vLLM enhanced answers) and 9 (evaluation) in a fresh Python process
# Use VLLM_WORKER_MULTIPROC_METHOD=spawn to avoid CUDA fork issues
echo "--- Running stages 8, 9 in fresh subprocess (vLLM + evaluation) ---"
VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_cross_query_se.py \
    --datasets nq webqa triviaqa hotpotqa squad \
    --output_dir "${OUTPUT_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --seeds 0 1 2 \
    --dev_size 500 \
    --top_k 5 \
    --top_k_dual 5 \
    --vllm_tp 4 \
    --gpu_id 0 \
    --stages 8 9

echo "=== Done ==="
ls -lh "${RESULTS_DIR}/" 2>/dev/null || true
