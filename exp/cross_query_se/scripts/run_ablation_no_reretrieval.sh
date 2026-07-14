#!/bin/bash
# Full-scale run: w/o re-retrieval ablation on all 5 datasets, seeds 0/1/2.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

echo "=== Ablation: w/o Re-Retrieval ==="
nvidia-smi | head -5

BASE_DIR="cross_query_se/outputs/cross_query_se"

for DATASET in nq webqa triviaqa hotpotqa squad; do
  echo "--- Dataset: $DATASET (stages 3-7) ---"
  VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_ablation_no_reretrieval.py \
    --datasets $DATASET \
    --output_dir cross_query_se/outputs/ablation_no_reretrieval \
    --base_output_dir "$BASE_DIR" \
    --results_dir cross_query_se/results/ablation_no_reretrieval \
    --seeds 0 1 2 \
    --dev_size 500 \
    --top_k 5 \
    --top_k_dual 5 \
    --vllm_tp 4 \
    --gpu_id 4 \
    --chunk_size 500000 \
    --query_batch_size 800 \
    --stages 3 4 5 51 7
  echo "--- Done stages 3-7: $DATASET ---"
done

echo "=== Copying enhanced_answers from base pipeline (AIS docs are identical) ==="
ABLATION_DIR="cross_query_se/outputs/ablation_no_reretrieval"
for DATASET in nq webqa triviaqa hotpotqa squad; do
  for SEED in 0 1 2; do
    SRC="$BASE_DIR/${DATASET}_enhanced_answers_seed${SEED}.jsonl"
    DST="$ABLATION_DIR/${DATASET}_enhanced_answers_seed${SEED}.jsonl"
    if [ -f "$SRC" ] && [ ! -f "$DST" ]; then
      cp "$SRC" "$DST"
      echo "Copied $SRC -> $DST"
    fi
  done
done

echo "=== Running stage 9 evaluation for all datasets ==="
for DATASET in nq webqa triviaqa hotpotqa squad; do
  echo "--- Dataset: $DATASET (stage 9) ---"
  VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_ablation_no_reretrieval.py \
    --datasets $DATASET \
    --output_dir cross_query_se/outputs/ablation_no_reretrieval \
    --base_output_dir "$BASE_DIR" \
    --results_dir cross_query_se/results/ablation_no_reretrieval \
    --seeds 0 1 2 \
    --dev_size 500 \
    --top_k 5 \
    --top_k_dual 5 \
    --vllm_tp 4 \
    --gpu_id 4 \
    --chunk_size 500000 \
    --query_batch_size 800 \
    --stages 9
  echo "--- Done stage 9: $DATASET ---"
done

echo "=== Ablation w/o Re-Retrieval Complete ==="
