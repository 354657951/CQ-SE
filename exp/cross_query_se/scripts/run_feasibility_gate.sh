#!/bin/bash
# Run feasibility gate: semantic filter + chunked GPU retrieval + V_ret computation.
# Uses ChunkedBGERetriever (torch matmul) instead of FAISS for compatibility.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"

source .venv/bin/activate
set -a; source .env; set +a

echo "=== Feasibility Gate: Filter + Retrieve + V_ret ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi | head -5

# chunk_size=500K: score matrix = 3000 queries * 500K * 4B = 6GB, chunk = 2GB => ~8GB total per pass
python cross_query_se/scripts/run_feasibility_gate.py \
    --datasets nq webqa triviaqa hotpotqa squad \
    --pert_dir cross_query_se/outputs/perturbations \
    --output_dir cross_query_se/outputs/feasibility_gate \
    --results_dir cross_query_se/results/feasibility_gate \
    --num_samples 500 \
    --k 5 \
    --top_k 5 \
    --tau 0.85 \
    --embeddings_path data/21MWiki_bge/corpus_embeddings.npy \
    --chunk_size 500000 \
    --gpu_id 0

echo "=== Done ==="
ls -lh cross_query_se/outputs/feasibility_gate/
ls -lh cross_query_se/results/feasibility_gate/
