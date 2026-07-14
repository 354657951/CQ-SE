#!/bin/bash
# Build FAISS indexes for 21MWiki and HotpotQA corpora using bge-large-en-v1.5.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"

source .venv/bin/activate
set -a; source .env; set +a

echo "=== Starting FAISS index build job ==="
echo "HF_HOME: $HF_HOME"
nvidia-smi

python cross_query_se/scripts/build_index.py --corpus both --batch_size 256

echo "=== Index build complete ==="
ls -lh data/21MWiki_bge/ data/hotpotqa_bge/
