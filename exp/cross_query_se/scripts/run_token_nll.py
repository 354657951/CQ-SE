# Compute per-query token-NLL (DTR uncertainty) for all datasets using vLLM.
# Token-NLL = mean(-log p) over generated answer tokens (excluding last EOS token).
# Mirrors the DTR/main.py uncertainty computation.
# Output: {ds}_token_nll.jsonl per dataset with keys: question, token_nll, answer

import os
import sys
import json
import math
import logging
import argparse
from typing import List, Dict

import numpy as np
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["nq", "webqa", "triviaqa", "hotpotqa", "squad"]
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using a single word or phrase."
)


def _jsonl_load(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _jsonl_save(records: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def compute_token_nll(probs: List[float]) -> float:
    """Compute mean token NLL, excluding last token (assumed EOS)."""
    if not probs:
        return 0.0
    nll_tokens = [(-math.log(p) if p and p > 0 else 0.0) for p in probs]
    # Exclude last token (EOS)
    if len(nll_tokens) > 1:
        nll_tokens = nll_tokens[:-1]
    return float(np.mean(nll_tokens))


def run_token_nll(args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.vllm_tp,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        top_p=1.0,
        top_k=1,
        max_tokens=64,
        stop=["<|im_end|>"],
        logprobs=1,
    )

    for ds in args.datasets:
        out_path = os.path.join(args.output_dir, f"{ds}_token_nll.jsonl")
        if os.path.exists(out_path):
            logger.info(f"[{ds}] Already exists, skipping: {out_path}")
            continue

        vllm_base_path = os.path.join(args.base_dir, f"{ds}_vllm_base.jsonl")
        if not os.path.exists(vllm_base_path):
            logger.warning(f"[{ds}] base file missing: {vllm_base_path}")
            continue

        records = _jsonl_load(vllm_base_path)
        # Use dev_size offset to get test set (consistent with main pipeline)
        test_records = records[args.dev_size:]
        if args.num_samples and args.num_samples < len(test_records):
            test_records = test_records[: args.num_samples]

        logger.info(f"[{ds}] Running token-NLL on {len(test_records)} examples")

        # Build prompts
        prompts = []
        for r in test_records:
            q = r["question"]
            msg = [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"Question: {q}"},
            ]
            prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

        # Run vLLM in batches
        results = []
        batch_size = args.batch_size
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start: start + batch_size]
            batch_records = test_records[start: start + batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            for rec, out in zip(batch_records, outputs):
                gen = out.outputs[0]
                answer_text = gen.text.strip()
                probs = []
                if gen.logprobs:
                    for token_logprobs in gen.logprobs:
                        if token_logprobs:
                            lp_obj = next(iter(token_logprobs.values()))
                            probs.append(math.exp(lp_obj.logprob))
                        else:
                            probs.append(0.0)
                nll = compute_token_nll(probs)
                results.append({
                    "question": rec["question"],
                    "token_nll": nll,
                    "answer": answer_text,
                })

        _jsonl_save(results, out_path)
        logger.info(f"[{ds}] Saved {len(results)} token-NLL records to {out_path}")
        avg_nll = float(np.mean([r["token_nll"] for r in results]))
        logger.info(f"[{ds}] Mean token-NLL: {avg_nll:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--base_dir", default="cross_query_se/outputs/cross_query_se")
    parser.add_argument("--output_dir", default="cross_query_se/outputs/token_nll")
    parser.add_argument("--dev_size", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=None, help="Limit test examples")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--vllm_tp", type=int, default=4)
    args = parser.parse_args()
    run_token_nll(args)


if __name__ == "__main__":
    main()
