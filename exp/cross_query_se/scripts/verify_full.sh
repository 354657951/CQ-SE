#!/bin/bash
# Full environment verification: CUDA, BGE, DeBERTa NLI, vLLM tokenizer
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_DIR="$(dirname "$EXP_DIR")"

source "$EXP_DIR/.venv/bin/activate"
# Use HF_HOME from environment if set, otherwise derive from workspace layout
export HF_HOME="${HF_HOME:-$WORKSPACE_DIR/hf_cache}"
export HF_HUB_DISABLE_XET=1

python3 2>/dev/null -c "
import os, sys, glob
import torch
import numpy as np
import faiss
from FlagEmbedding import FlagModel
from transformers import pipeline, AutoTokenizer
from sklearn.cluster import AgglomerativeClustering

hf_home = os.environ['HF_HOME']
hub = os.path.join(hf_home, 'hub')

def snap(model_id):
    d = model_id.replace('/', '--')
    paths = sorted(glob.glob(os.path.join(hub, f'models--{d}', 'snapshots', '*')))
    assert paths, f'No snapshot found for {model_id}'
    return paths[0]

print('=== GPU Environment Verification ===')
print(f'CUDA available:    {torch.cuda.is_available()}')
print(f'GPU count:         {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GB)')

# 1. FAISS (CPU) index
print()
print('[1] FAISS index ...')
idx = faiss.IndexFlatIP(1024)
vecs = np.random.rand(50, 1024).astype(np.float32)
idx.add(vecs)
D, I = idx.search(vecs[:3], 5)
assert I.shape == (3, 5), f'unexpected shape {I.shape}'
print(f'    FAISS IndexFlatIP(1024): add 50, search 3 -> {I.shape}  PASSED')

# 2. BGE embedding (GPU)
print()
print('[2] BGE bge-large-en-v1.5 ...')
bge_path = snap('BAAI/bge-large-en-v1.5')
model = FlagModel(
    bge_path,
    query_instruction_for_retrieval='Represent this sentence for searching relevant passages:',
    use_fp16=torch.cuda.is_available(),
)
sentences = ['What is the capital of France?', 'Who invented the telephone?', 'When did WWII end?']
embs = model.encode(sentences)
assert embs.shape == (3, 1024), f'unexpected shape {embs.shape}'
print(f'    BGE embeddings shape: {embs.shape}  PASSED')

# 3. DeBERTa NLI (bidirectional entailment)
print()
print('[3] DeBERTa-v2-xlarge-mnli NLI ...')
deberta_path = snap('microsoft/deberta-v2-xlarge-mnli')
nli = pipeline(
    'text-classification',
    model=deberta_path,
    device=0 if torch.cuda.is_available() else -1,
)
pairs = [
    ('Paris is the capital of France.', 'France has a capital city.'),
    ('The sky is blue.', 'The sky is green.'),
]
for hyp, prem in pairs:
    out = nli({'text': hyp, 'text_pair': prem})
    print(f'    [{out[\"label\"]:15s} {out[\"score\"]:.3f}]  \"{hyp[:40]}\" | \"{prem[:40]}\"')
print('    DeBERTa NLI  PASSED')

# 4. Agglomerative clustering (answer clustering)
print()
print('[4] Agglomerative clustering ...')
X = np.random.rand(10, 1024).astype(np.float32)
cl = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='cosine', linkage='average')
labels = cl.fit_predict(X)
print(f'    10 answers -> {len(set(labels))} clusters  PASSED')

# 5. Transformers tokenizer for Qwen2.5-7B
print()
print('[5] Qwen2.5-7B-Instruct tokenizer ...')
qwen7b_path = snap('Qwen/Qwen2.5-7B-Instruct')
tok = AutoTokenizer.from_pretrained(qwen7b_path, trust_remote_code=True)
ids = tok.encode('Hello, world!')
print(f'    Token IDs: {ids}  PASSED')

print()
print('=== ALL CHECKS PASSED ===')
"
