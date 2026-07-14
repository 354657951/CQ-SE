#!/bin/bash
# Full-scale run: tau sweep ablation.
# Runs tau in {0.70, 0.75, 0.80, 0.85, 0.90, 0.95} x 5 datasets, 1000 examples, seeds 0/1/2.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

echo "=== Ablation: Filter Threshold (tau) Sweep ==="
nvidia-smi | head -5

BASE_DIR="cross_query_se/outputs/cross_query_se"

for TAU in 0.70 0.75 0.80 0.85 0.90 0.95; do
  echo "=== tau=$TAU (stages 2-7) ==="
  TAU_STR=$(echo $TAU | tr '.' '_')
  ABLATION_DIR="cross_query_se/outputs/ablation_tau_${TAU_STR}"

  for DATASET in nq webqa triviaqa hotpotqa squad; do
    echo "--- Dataset: $DATASET, tau=$TAU ---"
    VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_ablation_tau_sweep.py \
      --datasets $DATASET \
      --tau $TAU \
      --base_output_dir "$BASE_DIR" \
      --results_dir cross_query_se/results/ablation_tau_sweep \
      --seeds 0 1 2 \
      --dev_size 500 \
      --max_examples 1000 \
      --top_k 5 \
      --top_k_dual 5 \
      --vllm_tp 4 \
      --gpu_id 4 \
      --chunk_size 500000 \
      --query_batch_size 800 \
      --stages 2 3 4 5 51 7
    echo "--- Done stages 2-7: $DATASET tau=$TAU ---"
  done

  echo "=== Copying enhanced_answers from base pipeline for tau=$TAU ==="
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

  echo "=== Running stage 9 for tau=$TAU ==="
  for DATASET in nq webqa triviaqa hotpotqa squad; do
    VLLM_WORKER_MULTIPROC_METHOD=spawn python cross_query_se/scripts/run_ablation_tau_sweep.py \
      --datasets $DATASET \
      --tau $TAU \
      --base_output_dir "$BASE_DIR" \
      --results_dir cross_query_se/results/ablation_tau_sweep \
      --seeds 0 1 2 \
      --dev_size 500 \
      --max_examples 1000 \
      --top_k 5 \
      --top_k_dual 5 \
      --vllm_tp 4 \
      --gpu_id 4 \
      --chunk_size 500000 \
      --query_batch_size 800 \
      --stages 9
    echo "--- Done stage 9: $DATASET tau=$TAU ---"
  done
done

echo "=== Ablation tau Sweep Complete ==="
