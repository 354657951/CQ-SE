#!/bin/bash
# Optimization run for 72B cross-query SE.
# Bug fixed: no_retrieval branch was using direct_answer (no context).
# For 72B, majority_vote over RAG answers is much better than direct answer even when H_cq is low.
# NQ: direct=0.410 vs majority=0.682; TriviaQA: 0.760 vs 0.847; SQuAD: 0.352 vs 0.646.
# Fix: all decision branches (no_retrieval, single_retrieval, enhanced_retrieval) use majority_answer
# or enhanced_answer, never direct_answer.
#
# Strategy: all stages 1-8 are already cached; just re-run stages 5b (enrich majority_answer) and 9 (eval).
# Stage 9 is CPU-only. No GPU needed.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

OUTPUT_DIR="cross_query_se/outputs/cross_query_se_72b"
RESULTS_DIR="cross_query_se/results/cross_query_se_72b_opt"

echo "=== Cross-Query SE 72B Optimization: Stages 5b, 9 ==="
echo "HF_HOME: $HF_HOME"

echo "--- Running stage 5b (enrich hcq files with majority_answer) and 9 (re-evaluate with fix) ---"
python cross_query_se/scripts/run_cross_query_se.py \
    --model Qwen/Qwen2.5-72B-Instruct \
    --datasets nq webqa triviaqa hotpotqa squad \
    --output_dir "${OUTPUT_DIR}" \
    --results_dir "${RESULTS_DIR}" \
    --seeds 0 1 2 \
    --dev_size 500 \
    --top_k 5 \
    --top_k_dual 5 \
    --vllm_tp 4 \
    --gpu_id 0 \
    --stages 51 9

echo "=== Done ==="
ls -lh "${RESULTS_DIR}/" 2>/dev/null || true
