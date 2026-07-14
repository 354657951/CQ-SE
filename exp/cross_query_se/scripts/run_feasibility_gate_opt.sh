#!/bin/bash
# Optimized feasibility gate: cosine-only filter (tau=0.80) + chunked GPU retrieval + V_ret computation.
# Improvement over original: removes DeBERTa NLI stage which over-rejects question paraphrases.
# Uses cached raw perturbations from cross_query_se/outputs/perturbations/.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a
echo "=== Optimized Feasibility Gate: Cosine-Only Filter (tau=0.80) ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5
python cross_query_se/scripts/run_feasibility_gate.py \
  --datasets nq webqa triviaqa hotpotqa squad \
  --pert_dir cross_query_se/outputs/perturbations \
  --output_dir cross_query_se/outputs/feasibility_gate_opt \
  --results_dir cross_query_se/results/feasibility_gate_opt \
  --num_samples 500 \
  --k 5 \
  --top_k 5 \
  --tau 0.80 \
  --cosine_only \
  --embeddings_path data/21MWiki_bge/corpus_embeddings.npy \
  --chunk_size 500000 \
  --gpu_id 0
echo "=== Done ==="
ls -lh cross_query_se/outputs/feasibility_gate_opt/
ls -lh cross_query_se/results/feasibility_gate_opt/
