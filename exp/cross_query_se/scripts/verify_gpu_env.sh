#!/bin/bash
# GPU environment sanity check: tests torch CUDA, BGE embedding, FAISS index, DeBERTa NLI
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_DIR="$(dirname "$EXP_DIR")"

source "$EXP_DIR/.venv/bin/activate"
# Use HF_HOME from environment if set, otherwise derive from workspace layout
export HF_HOME="${HF_HOME:-$WORKSPACE_DIR/hf_cache}"
export HF_HUB_DISABLE_XET=1

echo "=== GPU Environment Verification ==="

python3 -c "
import os
import torch
import faiss
import numpy as np

print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')

# Test FAISS GPU index
print('\n--- FAISS GPU test ---')
if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, 128)
    vecs = np.random.rand(100, 128).astype(np.float32)
    index.add(vecs)
    D, I = index.search(vecs[:5], 3)
    print(f'FAISS GPU index: OK (100 vecs, query returned shape {I.shape})')
else:
    index = faiss.IndexFlatIP(128)
    vecs = np.random.rand(100, 128).astype(np.float32)
    index.add(vecs)
    D, I = index.search(vecs[:5], 3)
    print(f'FAISS CPU index: OK (100 vecs, query returned shape {I.shape})')

# Test BGE embedding model
print('\n--- BGE embedding test ---')
import glob
from FlagEmbedding import FlagModel
hf_home = os.environ['HF_HOME']
bge_snaps = sorted(glob.glob(os.path.join(hf_home, 'hub/models--BAAI--bge-large-en-v1.5/snapshots/*')))
print(f'BGE model path: {bge_snaps[0]}')
model = FlagModel(
    bge_snaps[0],
    query_instruction_for_retrieval='Represent this sentence for searching relevant passages:',
    use_fp16=torch.cuda.is_available(),
)
emb = model.encode(['What is the capital of France?'])
print(f'BGE embedding shape: {emb.shape}')

# Test DeBERTa NLI model
print('\n--- DeBERTa NLI test ---')
from transformers import pipeline
deberta_snaps = sorted(glob.glob(os.path.join(hf_home, 'hub/models--microsoft--deberta-v2-xlarge-mnli/snapshots/*')))
print(f'DeBERTa model path: {deberta_snaps[0]}')
nli_pipe = pipeline('text-classification', model=deberta_snaps[0], device=0 if torch.cuda.is_available() else -1)
result = nli_pipe({'text': 'The cat is on the mat.', 'text_pair': 'There is a cat.'})
print(f'NLI result: {result}')

print('\n=== All GPU checks passed ===')
"
