#!/bin/bash
# Debug script v4: test full scoring with diagnostics.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${EXP_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$EXP_DIR"
source .venv/bin/activate
set -a; source .env; set +a

echo "=== INTRYGUE Debug Test v4 ==="
nvidia-smi | head -3

python -c "
import os, sys, torch, gc, json, math
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

HF_HOME = os.environ.get('HF_HOME', None)
cache_dir = os.path.join(HF_HOME, 'hub') if HF_HOME else None
device = 'cuda:0'
model_name = 'Qwen/Qwen2.5-7B-Instruct'
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir=cache_dir,
    dtype=torch.float16,
    attn_implementation='eager',
    device_map=device,
)
model.eval()

from cross_query_se.uncertainty.intrygue import INTRYGUEScorer
heads = [(2, 22), (4, 11), (5, 26), (5, 20), (4, 9), (8, 25), (3, 11), (2, 15), (2, 0), (15, 20)]
scorer = INTRYGUEScorer(model, tokenizer, heads, device=device, max_seq_len=256)
print(f'Heads by layer: {scorer.heads_by_layer}')
print(f'Hooks registered: {len(scorer._hooks)}')

# Test with real NQ examples from SUGAR output
sugar_records = []
with open('cross_query_se/outputs/sugar_baseline/nq_vllm_pass1.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 10: break
        sugar_records.append(json.loads(line))

print()
print('Testing scoring on 10 NQ examples:')
for rec in sugar_records[:5]:
    q = rec['question']
    ans = rec.get('direct_answer', '') or ''
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': f'Question: {q} Answer using a single word or phrase.'}],
        tokenize=False, add_generation_prompt=True
    )
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(ans, add_special_tokens=False)
    print(f'  Q={q[:50]!r} ans={ans!r} (prompt_len={len(p_ids)}, ans_len={len(a_ids)})')
    
    # Check hook state
    scorer._clear_cache()
    p_ids_trunc = p_ids[-240:]
    full_ids = p_ids_trunc + a_ids
    prompt_len = len(p_ids_trunc)
    ids_tensor = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        out = model(input_ids=ids_tensor, output_attentions=True, use_cache=False)
    
    print(f'    Hook cache keys: {list(scorer._attn_cache.keys())}')
    if scorer._attn_cache:
        layer_ex = list(scorer._attn_cache.keys())[0]
        print(f'    attn[{layer_ex}] shape: {scorer._attn_cache[layer_ex].shape}')
    
    # Entropy check
    logits = out.logits[0].float().cpu()
    ids_cpu = torch.tensor(full_ids)
    entropies = []
    for k in range(len(a_ids)):
        lp = prompt_len - 1 + k
        if lp >= logits.shape[0]: break
        log_p = F.log_softmax(logits[lp], dim=-1)
        p = log_p.exp()
        H = float(torch.where(p > 0, -p * log_p, torch.zeros_like(p)).sum().item())
        if math.isnan(H) or math.isinf(H): H = 0.0
        top_p, top_id = p.max(dim=-1)
        top_tok = tokenizer.decode([top_id.item()])
        entropies.append(H)
    
    print(f'    entropies={[round(h,4) for h in entropies[:5]]}')
    
    # SinkRate check 
    scorer._clear_cache()
    with torch.no_grad():
        out2 = model(input_ids=ids_tensor, output_attentions=True, use_cache=False)
    sink_rates = []
    for k in range(len(a_ids)):
        t = prompt_len + k
        if t < 2: 
            sink_rates.append(0.0)
            continue
        pred_tok = ids_cpu[t-1].item()
        head_srs = []
        for layer_idx, head_list in scorer.heads_by_layer.items():
            if layer_idx not in scorer._attn_cache: continue
            la = scorer._attn_cache[layer_idx]
            for head_idx in head_list:
                if head_idx >= la.shape[1]: continue
                attn_row = la[0, head_idx, t, :]
                from cross_query_se.uncertainty.intrygue import _sink_rate_from_attn_row
                sr = _sink_rate_from_attn_row(attn_row, ids_cpu, t)
                head_srs.append(sr)
        sink_rates.append(float(sum(head_srs)/len(head_srs)) if head_srs else 0.0)
    print(f'    sink_rates={[round(s,4) for s in sink_rates[:5]]}')
    
    del out, out2
    scorer._clear_cache()
    torch.cuda.empty_cache()

print()
print('Testing full scorer.score_from_answer:')
for rec in sugar_records[:5]:
    q = rec['question']
    ans = rec.get('direct_answer', '') or ''
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': f'Question: {q} Answer using a single word or phrase.'}],
        tokenize=False, add_generation_prompt=True
    )
    scores = scorer.score_from_answer(prompt, ans)
    print(f'  Q={q[:40]!r} ans={ans!r} -> {scores}')

print('ALL DONE')
"

echo "=== Debug v4 Done ==="
