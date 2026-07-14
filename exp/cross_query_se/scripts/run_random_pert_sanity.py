# Random perturbation sanity check for mechanism hypothesis validation.
# Generates K=5 random perturbations (word-shuffle or word-replacement) per query,
# runs re-retrieval + vLLM answer generation + DeBERTa H_rand clustering,
# then evaluates AUROC using -H_rand as confidence.
# Expected: H_rand should be HIGH but correlate poorly with correctness (low AUROC).
# Mirrors stages 2-5 of run_cross_query_se.py but uses random (non-semantic) perturbations.

import os
import sys
import json
import gc
import math
import logging
import argparse
import random
import re
import string
from collections import defaultdict
from typing import List, Dict, Tuple, Set

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from DTR.dataset.load_data import load_test_qa
from DTR.evaluation.metrics import exact_match_score, f1_score
from cross_query_se.retrieval.bge_retriever import ChunkedBGERetriever
from cross_query_se.uncertainty.cross_query_se import CrossQuerySE
from cross_query_se.adaptive.cross_query_trigger import majority_vote_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["nq", "webqa", "triviaqa", "hotpotqa", "squad"]
EMBEDDINGS_PATH = "data/21MWiki_bge/corpus_embeddings.npy"
CORPUS_PATH = "data/21MWiki/psgs_w100.tsv"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
DEBERTA_MODEL = "microsoft/deberta-v2-xlarge-mnli"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
K_PERTURB = 5
STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "about", "from", "into", "through",
    "what", "when", "where", "who", "which", "why", "how",
}


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
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _em(pred: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(float(exact_match_score(pred, g)) for g in gold_answers)


def _f1(pred: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(f1_score(pred, g)[0] for g in gold_answers)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _build_vocabulary(all_questions: List[str]) -> List[str]:
    """Build a flat list of non-stopword words from questions for random replacement."""
    vocab = set()
    for q in all_questions:
        for w in _tokenize(q):
            if w not in STOPWORDS and len(w) > 2:
                vocab.add(w)
    return sorted(vocab)


def _random_word_shuffle(question: str, rng: random.Random) -> str:
    """Strategy A: randomly shuffle all tokens in the question."""
    words = question.split()
    if len(words) <= 1:
        return question
    rng.shuffle(words)
    return " ".join(words)


def _random_word_replace(question: str, rng: random.Random, vocab: List[str]) -> str:
    """Strategy B: replace ~50% of content words with random vocabulary words."""
    words = question.split()
    new_words = []
    for w in words:
        cleaned = w.lower().strip(string.punctuation)
        if cleaned not in STOPWORDS and len(cleaned) > 2 and rng.random() < 0.5 and vocab:
            replacement = rng.choice(vocab)
            # Preserve capitalization pattern
            if w[0].isupper():
                replacement = replacement.capitalize()
            new_words.append(replacement)
        else:
            new_words.append(w)
    return " ".join(new_words)


def generate_random_perturbations(question: str, k: int, rng: random.Random, vocab: List[str]) -> List[str]:
    """Generate k random perturbations alternating between shuffle and replace strategies."""
    perts = []
    for i in range(k):
        if i % 2 == 0:
            perts.append(_random_word_shuffle(question, rng))
        else:
            perts.append(_random_word_replace(question, rng, vocab))
    # Deduplicate (keep original if identical)
    seen = {question}
    unique_perts = []
    for p in perts:
        if p not in seen:
            unique_perts.append(p)
            seen.add(p)
        if len(unique_perts) == k:
            break
    # Pad if needed with additional shuffles
    attempts = 0
    while len(unique_perts) < k and attempts < 20:
        p = _random_word_shuffle(question, rng)
        if p not in seen:
            unique_perts.append(p)
            seen.add(p)
        attempts += 1
    return unique_perts[:k]


def _load_corpus_texts(needed_ids: Set[int], corpus_path: str) -> Dict[int, Dict]:
    logger.info(f"Scanning corpus for {len(needed_ids)} doc IDs...")
    result = {}
    import pandas as pd
    chunk_iter = pd.read_csv(corpus_path, sep="\t", chunksize=500_000)
    loaded = 0
    for chunk in chunk_iter:
        for _, row in chunk.iterrows():
            did = int(row["id"]) - 1
            if did in needed_ids:
                result[did] = {"title": str(row["title"]), "text": str(row["text"])}
                loaded += 1
        if loaded >= len(needed_ids):
            break
    logger.info(f"Loaded {loaded} doc texts")
    return result


def _build_prompt_with_docs(q: str, docs: List[Dict], tokenizer) -> str:
    context = "\n".join(f"Title: {d['title']}. Content: {d['text']}" for d in docs)
    msg = [{"role": "user", "content": f"Question: {q}\n\nContext: {context}\n\nAnswer the question based on the above context using a single word or phrase."}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def _build_prompt_direct(q: str, tokenizer) -> str:
    msg = [{"role": "user", "content": f"Question: {q}\n\nAnswer the question using a single word or phrase."}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def _get_retrieval_scores(retriever, query_embs_t, flat_top_indices: List, top_k: int) -> List[float]:
    """Return max retrieval score for each query. If query_embs_t is None, returns 1.0 for all."""
    if query_embs_t is None:
        return [1.0 for _ in flat_top_indices]
    all_max_scores = []
    device = retriever.device
    for i, doc_set in enumerate(flat_top_indices):
        if not doc_set:
            all_max_scores.append(0.0)
            continue
        doc_ids = list(doc_set)[:top_k]
        doc_embs = np.array(retriever.corpus_embs[doc_ids], dtype=np.float32)
        doc_embs_t = torch.from_numpy(doc_embs).to(device)
        q_emb = query_embs_t[i: i + 1]
        scores = torch.matmul(q_emb, doc_embs_t.t()).squeeze(0)
        all_max_scores.append(float(scores.max().cpu()))
    return all_max_scores


def run_sanity(args):
    from sklearn.metrics import roc_auc_score

    # ── Load datasets ────────────────────────────────────────────────────────
    data = {}
    for ds in args.datasets:
        all_examples = load_test_qa(ds)
        test_examples = all_examples[args.dev_size:]
        if args.num_samples and args.num_samples < len(test_examples):
            test_examples = test_examples[: args.num_samples]
        data[ds] = test_examples
        logger.info(f"[{ds}] Loaded {len(test_examples)} test examples")

    all_questions_flat = [ex["question"] for ds in args.datasets for ex in data[ds]]
    vocab = _build_vocabulary(all_questions_flat)
    logger.info(f"Vocabulary size: {len(vocab)}")

    # ── Stage 1: Generate random perturbations ───────────────────────────────
    for ds in args.datasets:
        for seed in args.seeds:
            out_path = os.path.join(args.output_dir, f"{ds}_rand_perts_seed{seed}.jsonl")
            if os.path.exists(out_path):
                logger.info(f"[{ds}] seed={seed} Random perturbations exist, skipping")
                continue
            rng = random.Random(seed)
            records = []
            for ex in data[ds]:
                perts = generate_random_perturbations(ex["question"], K_PERTURB, rng, vocab)
                records.append({
                    "question": ex["question"],
                    "answers": ex["answers"],
                    "random_perturbations": perts,
                })
            _jsonl_save(records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(records)} random perturbation records")

    # ── Stage 2: BGE retrieval for random perturbations ──────────────────────
    # Process ALL seeds for a given dataset in ONE corpus pass (avoids repeated full-corpus scans).
    logger.info("=== Stage 2: BGE retrieval for random perturbations ===")
    device = f"cuda:{args.gpu_id}"
    retriever = ChunkedBGERetriever(
        embeddings_path=EMBEDDINGS_PATH,
        model_name=BGE_MODEL,
        device=device,
        chunk_size=args.chunk_size,
    )

    for ds in args.datasets:
        # Check which seeds need retrieval
        pending_seeds = [
            seed for seed in args.seeds
            if not os.path.exists(os.path.join(args.output_dir, f"{ds}_rand_ret_seed{seed}.jsonl"))
        ]
        if not pending_seeds:
            logger.info(f"[{ds}] All seeds done, skipping retrieval")
            continue

        # Load pert records for all pending seeds
        all_seed_pert_records = {}
        all_seed_query_lists = {}
        for seed in pending_seeds:
            records = _jsonl_load(os.path.join(args.output_dir, f"{ds}_rand_perts_seed{seed}.jsonl"))
            all_seed_pert_records[seed] = records
            all_seed_query_lists[seed] = [[r["question"]] + r["random_perturbations"] for r in records]

        # Build mega flat query list: all seeds concatenated
        mega_flat_queries = []
        mega_seed_lengths = {}  # seed -> number of flat queries for that seed
        for seed in pending_seeds:
            seed_flat = [q for qs in all_seed_query_lists[seed] for q in qs]
            mega_seed_lengths[seed] = len(seed_flat)
            mega_flat_queries.extend(seed_flat)

        logger.info(f"[{ds}] Retrieving {len(mega_flat_queries)} queries for seeds {pending_seeds} in ONE corpus pass...")

        # First encode ALL queries (efficient batch encoding)
        logger.info(f"[{ds}] Encoding {len(mega_flat_queries)} queries...")
        all_embs_t = retriever.encode_queries(mega_flat_queries)  # [Q, D]
        Q = all_embs_t.shape[0]

        # Single pass over corpus: for each corpus chunk, process ALL query sub-batches
        # This ensures corpus is loaded only once (not once per query batch)
        query_batch = args.query_batch_size
        top_scores = torch.full((Q, args.top_k), float("-inf"), device=retriever.device)
        top_indices = torch.zeros((Q, args.top_k), dtype=torch.long, device=retriever.device)

        n_chunks = (retriever.n_docs + args.chunk_size - 1) // args.chunk_size
        for chunk_idx in range(n_chunks):
            start = chunk_idx * args.chunk_size
            end = min(start + args.chunk_size, retriever.n_docs)
            # Load corpus chunk once
            chunk = torch.from_numpy(
                np.array(retriever.corpus_embs[start:end], dtype=np.float32)
            ).to(retriever.device)  # [C, D]

            # Process all queries in sub-batches against this corpus chunk
            for qb_start in range(0, Q, query_batch):
                qb_end = min(qb_start + query_batch, Q)
                q_sub = all_embs_t[qb_start:qb_end]  # [qb, D]
                scores_sub = torch.matmul(q_sub, chunk.t())  # [qb, C]

                combined_scores = torch.cat([top_scores[qb_start:qb_end], scores_sub], dim=1)
                combined_indices = torch.cat([
                    top_indices[qb_start:qb_end],
                    torch.arange(start, end, device=retriever.device).unsqueeze(0).expand(qb_end - qb_start, -1)
                ], dim=1)
                new_top_scores, sel = combined_scores.topk(args.top_k, dim=1)
                top_scores[qb_start:qb_end] = new_top_scores
                top_indices[qb_start:qb_end] = combined_indices.gather(1, sel)
                del scores_sub, combined_scores, combined_indices
            del chunk
            torch.cuda.empty_cache()
            if (chunk_idx + 1) % 2 == 0 or chunk_idx == n_chunks - 1:
                logger.info(f"[{ds}] Processed {end}/{retriever.n_docs} docs ({100*end/retriever.n_docs:.1f}%)")

        top_indices_cpu = top_indices.cpu().numpy()
        mega_top_indices = [set(int(i) for i in row) for row in top_indices_cpu]
        del top_scores, top_indices, all_embs_t
        torch.cuda.empty_cache()

        mega_scores = _get_retrieval_scores(retriever, None, mega_top_indices, args.top_k)

        # Distribute results back to per-seed records
        global_offset = 0
        for seed in pending_seeds:
            seed_query_lists = all_seed_query_lists[seed]
            pert_records = all_seed_pert_records[seed]
            n_seed_flat = mega_seed_lengths[seed]
            seed_top_indices = mega_top_indices[global_offset: global_offset + n_seed_flat]
            seed_scores = mega_scores[global_offset: global_offset + n_seed_flat]
            global_offset += n_seed_flat

            offset = 0
            ret_records = []
            for i, pr in enumerate(pert_records):
                l = len(seed_query_lists[i])
                per_query_doc_ids = [sorted(list(s)) for s in seed_top_indices[offset: offset + l]]
                per_query_max_scores = seed_scores[offset: offset + l]
                ret_records.append({
                    "question": pr["question"],
                    "answers": pr["answers"],
                    "random_perturbations": pr["random_perturbations"],
                    "per_query_doc_ids": per_query_doc_ids,
                    "per_query_max_scores": per_query_max_scores,
                })
                offset += l

            out_path = os.path.join(args.output_dir, f"{ds}_rand_ret_seed{seed}.jsonl")
            _jsonl_save(ret_records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(ret_records)} random retrieval records")

    del retriever
    torch.cuda.empty_cache()
    gc.collect()

    # ── Stage 3: vLLM per-perturbation answer generation ─────────────────────
    logger.info("=== Stage 3: vLLM per-perturbation answers ===")
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    from transformers import AutoTokenizer

    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.vllm_tp,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        max_model_len=4096,
        download_dir=cache_dir,
    )
    greedy_params = SamplingParams(n=1, temperature=0, top_p=1.0, max_tokens=64, stop=["<|im_end|>"])

    for ds in args.datasets:
        # Collect all needed doc IDs across seeds
        all_doc_ids = set()
        for seed in args.seeds:
            for r in _jsonl_load(os.path.join(args.output_dir, f"{ds}_rand_ret_seed{seed}.jsonl")):
                for doc_ids in r["per_query_doc_ids"]:
                    all_doc_ids.update(doc_ids)
        corpus_texts = _load_corpus_texts(all_doc_ids, CORPUS_PATH) if all_doc_ids else {}

        for seed in args.seeds:
            out_path = os.path.join(args.output_dir, f"{ds}_rand_vllm_seed{seed}.jsonl")
            if os.path.exists(out_path):
                logger.info(f"[{ds}] seed={seed} Random vLLM answers exist, skipping")
                continue

            ret_records = _jsonl_load(os.path.join(args.output_dir, f"{ds}_rand_ret_seed{seed}.jsonl"))

            all_prompts = []
            all_prompt_meta = []
            for i, pr in enumerate(ret_records):
                all_queries = [pr["question"]] + pr["random_perturbations"]
                for j, (q, doc_ids) in enumerate(zip(all_queries, pr["per_query_doc_ids"])):
                    docs = [corpus_texts[did] for did in doc_ids[: args.top_k] if did in corpus_texts]
                    if docs:
                        all_prompts.append(_build_prompt_with_docs(q, docs, tokenizer))
                    else:
                        all_prompts.append(_build_prompt_direct(q, tokenizer))
                    all_prompt_meta.append((i, j))

            logger.info(f"[{ds}] seed={seed} Generating {len(all_prompts)} answers...")
            all_outputs = llm.generate(all_prompts, greedy_params)
            all_answers_flat = [o.outputs[0].text.strip() for o in all_outputs]

            answers_by_example = defaultdict(dict)
            for (ex_idx, q_idx), ans in zip(all_prompt_meta, all_answers_flat):
                answers_by_example[ex_idx][q_idx] = ans

            vllm_records = []
            for i, pr in enumerate(ret_records):
                n_qs = 1 + len(pr["random_perturbations"])
                per_pert_answers = [answers_by_example[i].get(j, "") for j in range(n_qs)]
                vllm_records.append({
                    "question": pr["question"],
                    "answers": pr["answers"],
                    "random_perturbations": pr["random_perturbations"],
                    "per_pert_answers": per_pert_answers,
                    "per_query_max_scores": pr["per_query_max_scores"],
                })
            _jsonl_save(vllm_records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(vllm_records)} random vLLM records")

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # ── Stage 4: DeBERTa H_rand computation ──────────────────────────────────
    logger.info("=== Stage 4: DeBERTa H_rand computation ===")
    device = f"cuda:{args.gpu_id}"
    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    cq_estimator = CrossQuerySE(model_name=DEBERTA_MODEL, device=device, batch_size=64, cache_dir=cache_dir)

    for ds in args.datasets:
        for seed in args.seeds:
            out_path = os.path.join(args.output_dir, f"{ds}_rand_hcq_seed{seed}.jsonl")
            if os.path.exists(out_path):
                logger.info(f"[{ds}] seed={seed} H_rand exists, skipping")
                continue

            vllm_records = _jsonl_load(os.path.join(args.output_dir, f"{ds}_rand_vllm_seed{seed}.jsonl"))
            all_answers = [r["per_pert_answers"] for r in vllm_records]
            logger.info(f"[{ds}] seed={seed} Computing H_rand for {len(all_answers)} queries...")
            results = cq_estimator.compute_cross_query_se_batch(all_answers)

            records = []
            for pr, (hrand, cluster_ids) in zip(vllm_records, results):
                majority_ans = majority_vote_answer(pr["per_pert_answers"])
                records.append({
                    "question": pr["question"],
                    "answers": pr["answers"],
                    "hrand_score": hrand,
                    "cluster_ids": cluster_ids,
                    "per_pert_answers": pr["per_pert_answers"],
                    "majority_answer": majority_ans,
                })
            _jsonl_save(records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(records)} H_rand records")

    del cq_estimator
    torch.cuda.empty_cache()
    gc.collect()

    # ── Stage 5: Evaluate AUROC ───────────────────────────────────────────────
    logger.info("=== Stage 5: Evaluation ===")
    all_results = {}
    for ds in args.datasets:
        seed_metrics = []
        for seed in args.seeds:
            hrand_records = _jsonl_load(os.path.join(args.output_dir, f"{ds}_rand_hcq_seed{seed}.jsonl"))
            em_list, f1_list, hrand_list, correct_list = [], [], [], []
            for rec in hrand_records:
                pred = rec["majority_answer"]
                answers = rec["answers"]
                em = _em(pred, answers)
                f1 = _f1(pred, answers)
                hrand = rec["hrand_score"]
                em_list.append(em)
                f1_list.append(f1)
                hrand_list.append(hrand)
                correct_list.append(int(em > 0))

            try:
                auroc = float(roc_auc_score(correct_list, [-h for h in hrand_list]))
            except Exception:
                auroc = float("nan")

            seed_metrics.append({
                "seed": seed,
                "em": float(np.mean(em_list)),
                "f1": float(np.mean(f1_list)),
                "auroc": auroc,
                "n": len(em_list),
            })
            logger.info(f"[{ds}] seed={seed} EM={np.mean(em_list):.4f} F1={np.mean(f1_list):.4f} AUROC={auroc:.4f}")

        all_results[ds] = {
            "em_mean": float(np.mean([m["em"] for m in seed_metrics])),
            "em_std": float(np.std([m["em"] for m in seed_metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in seed_metrics])),
            "f1_std": float(np.std([m["f1"] for m in seed_metrics])),
            "auroc_mean": float(np.nanmean([m["auroc"] for m in seed_metrics])),
            "auroc_std": float(np.nanstd([m["auroc"] for m in seed_metrics])),
            "n_test": seed_metrics[0]["n"] if seed_metrics else 0,
            "seed_metrics": seed_metrics,
        }

    results_path = os.path.join(args.results_dir, "random_pert_sanity_results.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Summary
    logger.info("=== Summary ===")
    for ds, m in all_results.items():
        logger.info(f"  {ds}: EM={m['em_mean']:.4f} F1={m['f1_mean']:.4f} AUROC={m['auroc_mean']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--output_dir", default="cross_query_se/outputs/random_pert_sanity")
    parser.add_argument("--results_dir", default="cross_query_se/results/random_pert_sanity")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dev_size", type=int, default=500)
    parser.add_argument("--num_samples", type=int, default=1000, help="Test examples per dataset")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--vllm_tp", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=1_000_000)
    parser.add_argument("--query_batch_size", type=int, default=800,
                        help="Max queries per GPU batch during retrieval to avoid OOM")
    args = parser.parse_args()
    run_sanity(args)


if __name__ == "__main__":
    main()
