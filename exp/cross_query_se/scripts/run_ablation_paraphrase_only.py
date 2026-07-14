# Ablation: paraphrase-only variant of cross-query SE.
# Identical to run_cross_query_se.py except Stage 2 uses a paraphrase-only
# prompt that generates surface paraphrases ONLY (no perspective reframes).
# This tests whether perspective reframes are necessary for capturing
# retrievability gaps (they change retrieval anchors more aggressively).
#
# Reuses from base_output_dir: orig_retrieval, vllm_base, retrieval_info.
# New stages run: 2 (paraphrase-only prompts), 3, 4, 5, 5b, 7, 8, 9.

import os
import sys
import json
import gc
import logging
import argparse
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from DTR.dataset.load_data import load_test_qa
from DTR.evaluation.metrics import normalize_answer, exact_match_score, f1_score
from cross_query_se.retrieval.bge_retriever import ChunkedBGERetriever
from cross_query_se.perturbation.filter import SemanticEquivalenceFilter
from cross_query_se.uncertainty.cross_query_se import CrossQuerySE
from cross_query_se.adaptive.cross_query_trigger import (
    apply_cq_trigger, tune_cq_thresholds, select_best_answer_by_relevance,
    majority_vote_answer,
)
from cross_query_se.adaptive._ais import select_topk_of_query_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["nq", "webqa", "triviaqa", "hotpotqa", "squad"]
EMBEDDINGS_PATH = "data/21MWiki_bge/corpus_embeddings.npy"
CORPUS_PATH = "data/21MWiki/psgs_w100.tsv"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
DEBERTA_MODEL = "microsoft/deberta-v2-xlarge-mnli"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Paraphrase-only prompts (no perspective reframes)
PARAPHRASE_SYSTEM = (
    "You are an expert at reformulating questions using surface-level paraphrases. "
    "Your task is to generate diverse rewrites that vary only in wording and structure. "
    "Each rewrite must:\n"
    "1. Preserve the original question's answer (semantics intact)\n"
    "2. Vary the surface form — use synonyms, different grammatical structures, or word order changes\n"
    "Do NOT use entity aliases, perspective shifts, appositions, or any change to retrieval-relevant terms."
)

PARAPHRASE_USER = (
    "Generate exactly {k} surface paraphrases of the following question.\n"
    "Use only synonym substitution, grammatical restructuring, and word order changes.\n"
    "Do NOT use perspective shifts, entity reframes, or alias substitutions.\n\n"
    "Original question: {question}\n\n"
    "Output ONLY a numbered list (1. ... 2. ... etc.), one rewrite per line. "
    "No explanations, no preamble."
)


# ─── helpers ────────────────────────────────────────────────────────────────

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


def _load_corpus_texts(needed_ids: set, corpus_path: str) -> Dict[int, Dict]:
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


def _build_prompt_direct(q: str, tokenizer) -> str:
    msg = [{"role": "user", "content": f"Question: {q}\n\nAnswer the question using a single word or phrase."}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def _build_prompt_with_docs(q: str, docs: List[Dict], tokenizer) -> str:
    context = "\n".join(f"Title: {d['title']}. Content: {d['text']}" for d in docs)
    msg = [{"role": "user", "content": f"Question: {q}\n\nContext: {context}\n\nAnswer the question based on the above context using a single word or phrase."}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def _parse_perturbations(text: str, k: int) -> List[str]:
    import re
    lines = text.strip().split("\n")
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned and len(cleaned) > 5:
            results.append(cleaned)
    return results[:k]


def _get_retrieval_scores_for_flat(retriever, query_embs_t: torch.Tensor,
                                    flat_top_indices: List, top_k: int) -> List[float]:
    all_max_scores = []
    device = retriever.device
    for i, doc_set in enumerate(flat_top_indices):
        if not doc_set:
            all_max_scores.append(0.0)
            continue
        doc_ids = list(doc_set)[:top_k]
        doc_embs = np.array(retriever.corpus_embs[doc_ids], dtype=np.float32)
        doc_embs_t = torch.from_numpy(doc_embs).to(device)
        q_emb = query_embs_t[i:i+1]
        scores = torch.matmul(q_emb, doc_embs_t.t()).squeeze(0)
        all_max_scores.append(float(scores.max().cpu()))
    return all_max_scores


# ─── Stage 2 (ablation): paraphrase-only perturbation generation ─────────────

def stage2_paraphrase_only(datasets, data, base_output_dir, output_dir, seeds, k_perturb,
                            top_k, top_k_dual, vllm_tp, filt_tau, gpu_id, model=None, max_examples=-1):
    """Generate paraphrase-only perturbations (no perspective reframes)."""
    logger.info("=== Stage 2 (ablation): Paraphrase-only perturbation generation ===")
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    from transformers import AutoTokenizer

    model = model or QWEN_MODEL
    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None

    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    llm = LLM(
        model=model,
        tensor_parallel_size=vllm_tp,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        max_model_len=4096,
        download_dir=cache_dir,
    )
    perturb_temp_by_seed = {0: 0.7, 1: 0.8, 2: 0.9}

    for ds in datasets:
        questions = [ex["question"] for ex in data[ds]]

        for seed in seeds:
            pert_raw_out = os.path.join(output_dir, f"{ds}_perturbations_raw_seed{seed}.jsonl")
            if os.path.exists(pert_raw_out):
                logger.info(f"[{ds}] seed={seed} Raw paraphrase-only perturbations already exist, skipping")
                continue

            temp = perturb_temp_by_seed.get(seed, 0.7)
            logger.info(f"[{ds}] seed={seed} Generating paraphrase-only perturbations with temp={temp}...")

            perturb_params = SamplingParams(
                n=1, temperature=temp, top_p=0.9, max_tokens=256, stop=["<|im_end|>"]
            )

            perturb_prompts = []
            for q in questions:
                content = PARAPHRASE_USER.format(k=k_perturb + 2, question=q)
                msg = [
                    {"role": "system", "content": PARAPHRASE_SYSTEM},
                    {"role": "user", "content": content},
                ]
                perturb_prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

            perturb_outputs = llm.generate(perturb_prompts, perturb_params)
            raw_perts_list = [
                _parse_perturbations(o.outputs[0].text.strip(), k_perturb + 2)
                for o in perturb_outputs
            ]

            raw_records = [{"question": q, "raw_perturbations": raw_perts}
                           for q, raw_perts in zip(questions, raw_perts_list)]
            _jsonl_save(raw_records, pert_raw_out)
            logger.info(f"[{ds}] seed={seed} Saved {len(raw_records)} raw paraphrase-only perturbation records")

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Loading SemanticEquivalenceFilter (cosine_only)...")
    filt = SemanticEquivalenceFilter(tau=filt_tau, cosine_only=True, device=f"cuda:{gpu_id}")

    for ds in datasets:
        for seed in seeds:
            pert_out = os.path.join(output_dir, f"{ds}_perturbations_seed{seed}.jsonl")
            if os.path.exists(pert_out):
                logger.info(f"[{ds}] seed={seed} Filtered paraphrase-only perturbations already exist, skipping")
                continue

            pert_raw_out = os.path.join(output_dir, f"{ds}_perturbations_raw_seed{seed}.jsonl")
            raw_records = _jsonl_load(pert_raw_out)

            pert_records = []
            for rec in raw_records:
                q = rec["question"]
                raw_perts = rec["raw_perturbations"]
                filtered, _ = filt.filter_perturbations(q, raw_perts)
                filtered = filtered[:k_perturb]
                pert_records.append({
                    "question": q,
                    "raw_perturbations": raw_perts,
                    "filtered_perturbations": filtered,
                    "n_filtered": len(filtered),
                })

            avg_filt = np.mean([r["n_filtered"] for r in pert_records])
            _jsonl_save(pert_records, pert_out)
            logger.info(f"[{ds}] seed={seed} Saved {len(pert_records)} pert records, avg_filtered={avg_filt:.2f}")

    del filt
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Stage 2 (paraphrase-only) done")


# ─── Stage 3: Per-perturbation BGE retrieval ─────────────────────────────────

def stage3_retrieval_perturbations(datasets, data, output_dir, embeddings_path,
                                    chunk_size, top_k, gpu_id, query_batch_size=800):
    logger.info("=== Stage 3: Per-perturbation BGE retrieval ===")
    device = f"cuda:{gpu_id}"
    retriever = ChunkedBGERetriever(
        embeddings_path=embeddings_path,
        model_name=BGE_MODEL,
        device=device,
        chunk_size=chunk_size,
    )

    for ds in datasets:
        pert_files = sorted([f for f in os.listdir(output_dir)
                      if f.startswith(f"{ds}_perturbations_seed") and f.endswith(".jsonl")])
        for pf in pert_files:
            seed_str = pf.replace(f"{ds}_perturbations_seed", "").replace(".jsonl", "")
            try:
                seed = int(seed_str)
            except ValueError:
                continue

            out_path = os.path.join(output_dir, f"{ds}_pert_retrieval_seed{seed}.jsonl")
            if os.path.exists(out_path):
                logger.info(f"[{ds}] seed={seed} Perturbation retrieval already done, skipping")
                continue

            pert_records = _jsonl_load(os.path.join(output_dir, f"{ds}_perturbations_seed{seed}.jsonl"))
            all_query_lists = [
                [r["question"]] + r["filtered_perturbations"]
                for r in pert_records
            ]
            total_queries = sum(len(qs) for qs in all_query_lists)
            logger.info(f"[{ds}] seed={seed} Retrieval for {total_queries} queries (batch_size={query_batch_size})...")

            all_flat_top_indices = []
            all_flat_scores = []

            batch_start = 0
            while batch_start < len(all_query_lists):
                batch_end = batch_start
                batch_total = 0
                while batch_end < len(all_query_lists):
                    batch_total += len(all_query_lists[batch_end])
                    if batch_total > query_batch_size and batch_end > batch_start:
                        break
                    batch_end += 1
                    if batch_total >= query_batch_size:
                        break

                batch_query_lists = all_query_lists[batch_start:batch_end]
                flat_queries = [q for qs in batch_query_lists for q in qs]

                logger.info(f"[{ds}] seed={seed} Encoding {len(flat_queries)} queries (examples {batch_start}-{batch_end})...")
                query_embs_t = retriever.encode_queries(flat_queries)
                batch_top_indices = retriever._chunked_topk(query_embs_t, top_k)
                batch_scores = _get_retrieval_scores_for_flat(retriever, query_embs_t, batch_top_indices, top_k)

                all_flat_top_indices.extend(batch_top_indices)
                all_flat_scores.extend(batch_scores)

                del query_embs_t
                torch.cuda.empty_cache()
                batch_start = batch_end

            results_grouped = []
            scores_grouped = []
            offset = 0
            for qs in all_query_lists:
                l = len(qs)
                results_grouped.append(all_flat_top_indices[offset: offset + l])
                scores_grouped.append(all_flat_scores[offset: offset + l])
                offset += l

            pert_ret_records = []
            for i, pr in enumerate(pert_records):
                per_pert_doc_ids = [sorted(list(s)) for s in results_grouped[i]]
                pert_ret_records.append({
                    "question": pr["question"],
                    "filtered_perturbations": pr["filtered_perturbations"],
                    "per_query_doc_ids": per_pert_doc_ids,
                    "per_query_max_scores": scores_grouped[i],
                })
            _jsonl_save(pert_ret_records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(pert_ret_records)} pert retrieval records")

    del retriever
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Stage 3 done")


# ─── Stage 4: vLLM pass 2 — per-perturbation greedy answers ─────────────────

def stage4_vllm_perturbation_answers(datasets, data, output_dir, seeds, top_k, vllm_tp, model=None):
    logger.info("=== Stage 4: vLLM pass 2 (per-perturbation greedy answers) ===")

    needs_generation = False
    for ds in datasets:
        for seed in seeds:
            if not os.path.exists(os.path.join(output_dir, f"{ds}_vllm_pert_seed{seed}.jsonl")):
                needs_generation = True
                break
        if needs_generation:
            break

    if not needs_generation:
        logger.info("All per-pert vLLM outputs already exist, skipping Stage 4")
        return

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    from transformers import AutoTokenizer

    model = model or QWEN_MODEL
    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None

    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    llm = LLM(
        model=model,
        tensor_parallel_size=vllm_tp,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        max_model_len=4096,
        download_dir=cache_dir,
    )
    greedy_params = SamplingParams(n=1, temperature=0, top_p=1.0, max_tokens=64, stop=["<|im_end|>"])

    for ds in datasets:
        seeds_needed = [s for s in seeds if not os.path.exists(
            os.path.join(output_dir, f"{ds}_vllm_pert_seed{s}.jsonl"))]
        if not seeds_needed:
            logger.info(f"[{ds}] All per-pert vLLM outputs exist, skipping")
            continue

        all_doc_ids = set()
        for seed in seeds_needed:
            pert_ret_path = os.path.join(output_dir, f"{ds}_pert_retrieval_seed{seed}.jsonl")
            if os.path.exists(pert_ret_path):
                for r in _jsonl_load(pert_ret_path):
                    for doc_ids in r["per_query_doc_ids"]:
                        all_doc_ids.update(doc_ids)
        corpus_texts = _load_corpus_texts(all_doc_ids, CORPUS_PATH) if all_doc_ids else {}

        for seed in seeds_needed:
            out_path = os.path.join(output_dir, f"{ds}_vllm_pert_seed{seed}.jsonl")
            pert_ret_records = _jsonl_load(
                os.path.join(output_dir, f"{ds}_pert_retrieval_seed{seed}.jsonl")
            )

            all_prompts = []
            all_prompt_meta = []
            for i, pr in enumerate(pert_ret_records):
                q_orig = pr["question"]
                perturbs = pr["filtered_perturbations"]
                all_queries = [q_orig] + perturbs
                doc_ids_list = pr["per_query_doc_ids"]

                for j, (q, doc_ids) in enumerate(zip(all_queries, doc_ids_list)):
                    docs = [corpus_texts[did] for did in doc_ids[:top_k] if did in corpus_texts]
                    if docs:
                        all_prompts.append(_build_prompt_with_docs(q, docs, tokenizer))
                    else:
                        all_prompts.append(_build_prompt_direct(q, tokenizer))
                    all_prompt_meta.append((i, j))

            logger.info(f"[{ds}] seed={seed} Generating {len(all_prompts)} per-perturbation greedy answers...")
            all_outputs = llm.generate(all_prompts, greedy_params)
            all_answers_flat = [o.outputs[0].text.strip() for o in all_outputs]

            answers_by_example = defaultdict(dict)
            for (ex_idx, q_idx), ans in zip(all_prompt_meta, all_answers_flat):
                answers_by_example[ex_idx][q_idx] = ans

            pert_vllm_records = []
            for i, pr in enumerate(pert_ret_records):
                n_qs = len([pr["question"]] + pr["filtered_perturbations"])
                per_pert_answers = [answers_by_example[i].get(j, "") for j in range(n_qs)]
                pert_vllm_records.append({
                    "question": pr["question"],
                    "filtered_perturbations": pr["filtered_perturbations"],
                    "per_pert_answers": per_pert_answers,
                    "per_query_max_scores": pr["per_query_max_scores"],
                })
            _jsonl_save(pert_vllm_records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(pert_vllm_records)} per-pert vLLM records")

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Stage 4 done")


# ─── Stage 5: DeBERTa H_cq computation ──────────────────────────────────────

def stage5_hcq_computation(datasets, data, output_dir, seeds, gpu_id):
    logger.info("=== Stage 5: DeBERTa H_cq computation ===")
    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    device = f"cuda:{gpu_id}"

    cq_estimator = CrossQuerySE(model_name=DEBERTA_MODEL, device=device, batch_size=64, cache_dir=cache_dir)

    for ds in datasets:
        for seed in seeds:
            out_path = os.path.join(output_dir, f"{ds}_hcq_seed{seed}.jsonl")
            if os.path.exists(out_path):
                logger.info(f"[{ds}] seed={seed} Stage 5 already done, skipping")
                continue

            pert_vllm = _jsonl_load(os.path.join(output_dir, f"{ds}_vllm_pert_seed{seed}.jsonl"))
            all_answers = [r["per_pert_answers"] for r in pert_vllm]
            all_max_scores = [r["per_query_max_scores"] for r in pert_vllm]

            logger.info(f"[{ds}] seed={seed} Computing H_cq for {len(all_answers)} queries...")
            results = cq_estimator.compute_cross_query_se_batch(all_answers)

            records = []
            for i, (pr, (hcq, cluster_ids)) in enumerate(zip(pert_vllm, results)):
                best_ans = select_best_answer_by_relevance(pr["per_pert_answers"], all_max_scores[i])
                majority_ans = majority_vote_answer(pr["per_pert_answers"])
                records.append({
                    "question": pr["question"],
                    "hcq_score": hcq,
                    "cluster_ids": cluster_ids,
                    "per_pert_answers": pr["per_pert_answers"],
                    "best_answer_by_relevance": best_ans,
                    "majority_answer": majority_ans,
                })
            _jsonl_save(records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(records)} H_cq records")

    del cq_estimator
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Stage 5 done")


def stage5b_enrich_hcq(datasets, output_dir, seeds):
    logger.info("=== Stage 5b: Enriching H_cq files with majority_answer ===")
    for ds in datasets:
        for seed in seeds:
            path = os.path.join(output_dir, f"{ds}_hcq_seed{seed}.jsonl")
            if not os.path.exists(path):
                continue
            records = _jsonl_load(path)
            if records and "majority_answer" in records[0]:
                continue
            updated = []
            for r in records:
                per_pert = r.get("per_pert_answers", [])
                r["majority_answer"] = majority_vote_answer(per_pert) if per_pert else ""
                updated.append(r)
            _jsonl_save(updated, path)
            logger.info(f"[{ds}] seed={seed} Enriched {len(updated)} hcq records with majority_answer")
    logger.info("Stage 5b done")


# ─── Stage 7: Threshold tuning + AIS doc selection ──────────────────────────

def stage7_threshold_and_ais(datasets, data, base_output_dir, output_dir, seeds, dev_size, top_k, top_k_dual,
                               embeddings_path, chunk_size, gpu_id):
    logger.info("=== Stage 7: Threshold tuning + AIS doc selection ===")
    thresholds_path = os.path.join(output_dir, "cq_thresholds.json")

    thresholds = {}
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            thresholds = json.load(f)

    for ds in datasets:
        if ds in thresholds:
            logger.info(f"[{ds}] Thresholds already tuned, skipping")
        else:
            logger.info(f"[{ds}] Tuning thresholds on dev split [0:{dev_size}]...")
            base_records = _jsonl_load(os.path.join(base_output_dir, f"{ds}_vllm_base.jsonl"))
            dev_base = {r["question"]: r for r in base_records[:dev_size]}

            avg_hcq_by_q = defaultdict(list)
            for seed in seeds:
                hcq_recs = _jsonl_load(os.path.join(output_dir, f"{ds}_hcq_seed{seed}.jsonl"))
                for r in hcq_recs[:dev_size]:
                    avg_hcq_by_q[r["question"]].append(r["hcq_score"])

            majority_by_q = {}
            for seed in seeds:
                hcq_recs_dev = _jsonl_load(os.path.join(output_dir, f"{ds}_hcq_seed{seed}.jsonl"))
                for r in hcq_recs_dev[:dev_size]:
                    ans = r.get("majority_answer") or r.get("best_answer_by_relevance", "")
                    if r["question"] not in majority_by_q:
                        majority_by_q[r["question"]] = []
                    majority_by_q[r["question"]].append(ans)

            dev_records = []
            for q, br in dev_base.items():
                avg_hcq = float(np.mean(avg_hcq_by_q[q])) if avg_hcq_by_q[q] else 0.0
                majority_ans = majority_by_q.get(q, [br["rag3_answer"]])
                from collections import Counter as _Counter
                majority_ans_final = _Counter(majority_ans).most_common(1)[0][0] if majority_ans else br["rag3_answer"]
                dev_records.append({
                    "hcq_score": avg_hcq,
                    "answers": br["answers"],
                    "direct_answer": br["direct_answer"],
                    "rag3_answer": br["rag3_answer"],
                    "majority_answer": majority_ans_final,
                    "enhanced_answer": None,
                })

            tau_low, tau_high = tune_cq_thresholds(dev_records)
            thresholds[ds] = {"tau_low": tau_low, "tau_high": tau_high}
            with open(thresholds_path, "w") as f:
                json.dump(thresholds, f, indent=2)
            logger.info(f"[{ds}] tau_low={tau_low}, tau_high={tau_high}")

    device = f"cuda:{gpu_id}"
    retriever = ChunkedBGERetriever(
        embeddings_path=embeddings_path,
        model_name=BGE_MODEL,
        device=device,
        chunk_size=chunk_size,
    )

    for ds in datasets:
        for seed in seeds:
            out_path = os.path.join(output_dir, f"{ds}_enhanced_docs_seed{seed}.jsonl")
            if os.path.exists(out_path):
                logger.info(f"[{ds}] seed={seed} AIS already done, skipping")
                continue

            tau_low = thresholds[ds]["tau_low"]
            tau_high = thresholds[ds]["tau_high"]

            base_records = _jsonl_load(os.path.join(base_output_dir, f"{ds}_vllm_base.jsonl"))
            hcq_records = _jsonl_load(os.path.join(output_dir, f"{ds}_hcq_seed{seed}.jsonl"))
            ret_query = {r["question"]: r for r in _jsonl_load(
                os.path.join(base_output_dir, f"{ds}_orig_retrieval.jsonl")
            )}
            ret_info = {r["question"]: r for r in _jsonl_load(
                os.path.join(base_output_dir, f"{ds}_retrieval_info.jsonl")
            )}

            test_base = base_records[dev_size:]
            test_hcq = hcq_records[dev_size:]

            enhanced_questions, enhanced_indices = [], []
            for i, (br, hr) in enumerate(zip(test_base, test_hcq)):
                assert br["question"] == hr["question"]
                if apply_cq_trigger(hr["hcq_score"], tau_low, tau_high) == "enhanced_retrieval":
                    enhanced_questions.append(br["question"])
                    enhanced_indices.append(i)

            logger.info(f"[{ds}] seed={seed} {len(enhanced_questions)} queries need enhanced retrieval")

            enhanced_docs_map = {}
            if enhanced_questions:
                batch_doc_ids_q, batch_doc_embs_q, batch_D_q, batch_query_embs = [], [], [], []
                batch_doc_ids_info, batch_doc_embs_info, batch_D_info, batch_info_embs = [], [], [], []
                for q in enhanced_questions:
                    rq = ret_query[q]
                    ri = ret_info[q]
                    batch_doc_ids_q.append(rq["doc_ids"][:top_k_dual])
                    batch_doc_embs_q.append(rq["doc_embs"][:top_k_dual])
                    batch_D_q.append(rq["scores"][:top_k_dual])
                    batch_query_embs.append(rq["query_emb"])
                    batch_doc_ids_info.append(ri["doc_ids_info"][:top_k_dual])
                    batch_doc_embs_info.append(ri["doc_embs_info"][:top_k_dual])
                    batch_D_info.append(ri["scores_info"][:top_k_dual])
                    batch_info_embs.append(ri["info_emb"])
                selected_doc_ids, _ = select_topk_of_query_info(
                    np.array(batch_doc_ids_q),
                    np.array(batch_doc_ids_info),
                    np.array(batch_doc_embs_q),
                    np.array(batch_doc_embs_info),
                    np.array(batch_D_q),
                    np.array(batch_D_info),
                    np.array(batch_query_embs),
                    np.array(batch_info_embs),
                    topk_new=top_k,
                    consider_adaptive=False,
                )
                for q, sel_ids in zip(enhanced_questions, selected_doc_ids):
                    enhanced_docs_map[q] = [int(i) for i in sel_ids]

            records = []
            for i, (br, hr) in enumerate(zip(test_base, test_hcq)):
                q = br["question"]
                decision = apply_cq_trigger(hr["hcq_score"], tau_low, tau_high)
                records.append({
                    "question": q,
                    "hcq_score": hr["hcq_score"],
                    "decision": decision,
                    "enhanced_doc_ids": enhanced_docs_map.get(q, []),
                })
            _jsonl_save(records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(records)} AIS records")

    del retriever
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Stage 7 done")


# ─── Stage 8: vLLM generation pass 3 (enhanced answers) ─────────────────────

def stage8_vllm_enhanced(datasets, data, base_output_dir, output_dir, seeds, dev_size, top_k, vllm_tp, model=None):
    logger.info("=== Stage 8: vLLM enhanced answer generation ===")

    needs_pass = False
    for ds in datasets:
        for seed in seeds:
            p = os.path.join(output_dir, f"{ds}_enhanced_docs_seed{seed}.jsonl")
            out_p = os.path.join(output_dir, f"{ds}_enhanced_answers_seed{seed}.jsonl")
            if os.path.exists(out_p):
                continue
            if os.path.exists(p):
                records = _jsonl_load(p)
                if any(r["decision"] == "enhanced_retrieval" for r in records):
                    needs_pass = True
                    break
        if needs_pass:
            break

    if not needs_pass:
        logger.info("No enhanced queries; skipping Stage 8")
        for ds in datasets:
            for seed in seeds:
                out_p = os.path.join(output_dir, f"{ds}_enhanced_answers_seed{seed}.jsonl")
                if not os.path.exists(out_p):
                    _jsonl_save([], out_p)
        return

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    from transformers import AutoTokenizer

    model = model or QWEN_MODEL
    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)
    llm = LLM(
        model=model,
        tensor_parallel_size=vllm_tp,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        max_model_len=4096,
        download_dir=cache_dir,
    )
    greedy_params = SamplingParams(n=1, temperature=0, top_p=1.0, max_tokens=64, stop=["<|im_end|>"])

    for ds in datasets:
        all_enhanced_doc_ids = set()
        for seed in seeds:
            ais_path = os.path.join(output_dir, f"{ds}_enhanced_docs_seed{seed}.jsonl")
            if os.path.exists(ais_path):
                for r in _jsonl_load(ais_path):
                    all_enhanced_doc_ids.update(r["enhanced_doc_ids"])
        corpus_texts = _load_corpus_texts(all_enhanced_doc_ids, CORPUS_PATH) if all_enhanced_doc_ids else {}

        for seed in seeds:
            out_path = os.path.join(output_dir, f"{ds}_enhanced_answers_seed{seed}.jsonl")
            if os.path.exists(out_path):
                logger.info(f"[{ds}] seed={seed} Stage 8 already done, skipping")
                continue

            ais_records = _jsonl_load(os.path.join(output_dir, f"{ds}_enhanced_docs_seed{seed}.jsonl"))
            enhanced_qs = [r["question"] for r in ais_records if r["decision"] == "enhanced_retrieval"]
            enhanced_doc_ids_map = {r["question"]: r["enhanced_doc_ids"] for r in ais_records}

            if not enhanced_qs:
                _jsonl_save([], out_path)
                continue

            logger.info(f"[{ds}] seed={seed} Generating enhanced answers for {len(enhanced_qs)} queries...")
            prompts = []
            for q in enhanced_qs:
                doc_ids = enhanced_doc_ids_map[q]
                docs = [corpus_texts[did] for did in doc_ids if did in corpus_texts]
                prompts.append(_build_prompt_with_docs(q, docs, tokenizer))

            outputs = llm.generate(prompts, greedy_params)
            answers = [o.outputs[0].text.strip() for o in outputs]

            records = [{"question": q, "enhanced_answer": ans} for q, ans in zip(enhanced_qs, answers)]
            _jsonl_save(records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(records)} enhanced answers")

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Stage 8 done")


# ─── Stage 9: Evaluation ─────────────────────────────────────────────────────

def stage9_evaluate(datasets, data, base_output_dir, output_dir, seeds, dev_size, top_k, results_dir):
    logger.info("=== Stage 9: Evaluation ===")
    from sklearn.metrics import roc_auc_score

    os.makedirs(results_dir, exist_ok=True)
    thresholds = json.load(open(os.path.join(output_dir, "cq_thresholds.json")))

    all_results = {}
    for ds in datasets:
        tau_low = thresholds[ds]["tau_low"]
        tau_high = thresholds[ds]["tau_high"]

        base_records = _jsonl_load(os.path.join(base_output_dir, f"{ds}_vllm_base.jsonl"))
        test_base = base_records[dev_size:]

        seed_metrics = []
        for seed in seeds:
            hcq_records = _jsonl_load(os.path.join(output_dir, f"{ds}_hcq_seed{seed}.jsonl"))
            test_hcq = hcq_records[dev_size:]
            test_hcq_by_q = {r["question"]: r for r in test_hcq}

            ais_records = _jsonl_load(os.path.join(output_dir, f"{ds}_enhanced_docs_seed{seed}.jsonl"))
            ais_by_q = {r["question"]: r for r in ais_records}

            enh_answers_by_q = {}
            enh_ans_path = os.path.join(output_dir, f"{ds}_enhanced_answers_seed{seed}.jsonl")
            if os.path.exists(enh_ans_path):
                for r in _jsonl_load(enh_ans_path):
                    enh_answers_by_q[r["question"]] = r["enhanced_answer"]

            em_list, f1_list, hcq_list, correct_list = [], [], [], []
            n_no_ret, n_single_ret, n_enhanced_ret = 0, 0, 0
            total_retriever_calls = 0
            total_docs_retrieved = 0

            for br in test_base:
                q = br["question"]
                answers = br["answers"]
                hcqr = test_hcq_by_q.get(q, {"hcq_score": 0.0, "per_pert_answers": [br["rag3_answer"]]})
                aisr = ais_by_q.get(q, {"decision": "single_retrieval"})

                hcq = hcqr["hcq_score"]
                decision = aisr["decision"]
                per_pert_ans = hcqr.get("per_pert_answers", [])
                majority_ans = majority_vote_answer(per_pert_ans) if per_pert_ans else br["rag3_answer"]

                if decision == "no_retrieval":
                    pred = majority_ans
                    n_no_ret += 1
                    retriever_calls = 1
                    docs_retrieved = 0
                elif decision == "single_retrieval":
                    pred = majority_ans
                    n_single_ret += 1
                    n_perts = len(per_pert_ans) - 1
                    retriever_calls = 1 + (1 + n_perts)
                    docs_retrieved = top_k
                else:
                    pred = enh_answers_by_q.get(q, majority_ans)
                    n_enhanced_ret += 1
                    n_perts = len(per_pert_ans) - 1
                    retriever_calls = 1 + (1 + n_perts) + 1
                    docs_retrieved = top_k

                em = _em(pred, answers)
                f1 = _f1(pred, answers)
                em_list.append(em)
                f1_list.append(f1)
                hcq_list.append(hcq)
                correct_list.append(int(em > 0))
                total_retriever_calls += retriever_calls
                total_docs_retrieved += docs_retrieved

            n_test = len(test_base)
            avg_em = float(np.mean(em_list)) if em_list else 0.0
            avg_f1 = float(np.mean(f1_list)) if f1_list else 0.0
            avg_ret_calls = total_retriever_calls / max(n_test, 1)
            avg_docs = total_docs_retrieved / max(n_test, 1)

            try:
                auroc = float(roc_auc_score(correct_list, [-h for h in hcq_list]))
            except Exception:
                auroc = float("nan")

            seed_metrics.append({
                "seed": seed,
                "em": avg_em,
                "f1": avg_f1,
                "auroc": auroc,
                "avg_retriever_calls": avg_ret_calls,
                "avg_docs_retrieved": avg_docs,
                "n_test": n_test,
                "n_no_retrieval": n_no_ret,
                "n_single_retrieval": n_single_ret,
                "n_enhanced_retrieval": n_enhanced_ret,
            })
            logger.info(
                f"[{ds}] seed={seed} EM={avg_em:.4f} F1={avg_f1:.4f} "
                f"AUROC={auroc:.4f} RetCalls={avg_ret_calls:.2f}"
            )

        em_vals = [m["em"] for m in seed_metrics]
        f1_vals = [m["f1"] for m in seed_metrics]
        auroc_vals = [m["auroc"] for m in seed_metrics if not math.isnan(m["auroc"])]
        rc_vals = [m["avg_retriever_calls"] for m in seed_metrics]
        n_test = seed_metrics[0]["n_test"] if seed_metrics else 0
        last = seed_metrics[-1] if seed_metrics else {}

        all_results[ds] = {
            "em_mean": float(np.mean(em_vals)),
            "em_std": float(np.std(em_vals)),
            "f1_mean": float(np.mean(f1_vals)),
            "f1_std": float(np.std(f1_vals)),
            "auroc_mean": float(np.mean(auroc_vals)) if auroc_vals else float("nan"),
            "auroc_std": float(np.std(auroc_vals)) if auroc_vals else float("nan"),
            "avg_retriever_calls_mean": float(np.mean(rc_vals)),
            "tau_low": tau_low,
            "tau_high": tau_high,
            "n_test": n_test,
            "retrieval_mode_pct": {
                "no_retrieval": round(100 * last.get("n_no_retrieval", 0) / max(n_test, 1), 1),
                "single_retrieval": round(100 * last.get("n_single_retrieval", 0) / max(n_test, 1), 1),
                "enhanced_retrieval": round(100 * last.get("n_enhanced_retrieval", 0) / max(n_test, 1), 1),
            },
            "per_seed": seed_metrics,
        }
        logger.info(
            f"[{ds}] FINAL EM={all_results[ds]['em_mean']:.4f}±{all_results[ds]['em_std']:.4f} "
            f"F1={all_results[ds]['f1_mean']:.4f}±{all_results[ds]['f1_std']:.4f} "
            f"AUROC={all_results[ds]['auroc_mean']:.4f}±{all_results[ds]['auroc_std']:.4f}"
        )

    summary_path = os.path.join(results_dir, "ablation_paraphrase_only_results.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                existing = json.load(f)
            existing.update(all_results)
            all_results = existing
        except Exception:
            pass
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {summary_path}")

    print("\n=== ABLATION: PARAPHRASE-ONLY RESULTS ===")
    print(f"{'Dataset':<12} {'EM':>10} {'F1':>10} {'AUROC':>10} {'RetCalls':>10}")
    print("-" * 55)
    for ds, r in all_results.items():
        print(
            f"{ds:<12} {r['em_mean']:>7.4f}±{r['em_std']:.3f} "
            f"{r['f1_mean']:>7.4f}±{r['f1_std']:.3f} "
            f"{r['auroc_mean']:>7.4f}±{r['auroc_std']:.3f} "
            f"{r['avg_retriever_calls_mean']:>8.2f}"
        )
    return all_results


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--output_dir", default="cross_query_se/outputs/ablation_paraphrase_only")
    parser.add_argument("--base_output_dir", default="cross_query_se/outputs/cross_query_se",
                        help="Base pipeline output dir to reuse Stage 1/6 outputs from")
    parser.add_argument("--results_dir", default="cross_query_se/results/ablation_paraphrase_only")
    parser.add_argument("--k_perturb", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dev_size", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_k_dual", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=500_000)
    parser.add_argument("--vllm_tp", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--filt_tau", type=float, default=0.75)
    parser.add_argument("--query_batch_size", type=int, default=800)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--stages", nargs="+", type=int, default=[2, 3, 4, 5, 51, 7, 8, 9])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = {}
    for ds in args.datasets:
        examples = load_test_qa(ds, num_samples=args.max_examples)
        data[ds] = examples
        logger.info(f"Loaded {len(examples)} examples for {ds}")

    if 2 in args.stages:
        stage2_paraphrase_only(args.datasets, data, args.base_output_dir, args.output_dir,
                                args.seeds, args.k_perturb, args.top_k, args.top_k_dual,
                                args.vllm_tp, args.filt_tau, args.gpu_id, max_examples=args.max_examples)
    if 3 in args.stages:
        stage3_retrieval_perturbations(args.datasets, data, args.output_dir, EMBEDDINGS_PATH,
                                       args.chunk_size, args.top_k, args.gpu_id,
                                       query_batch_size=args.query_batch_size)
    if 4 in args.stages:
        stage4_vllm_perturbation_answers(args.datasets, data, args.output_dir, args.seeds,
                                          args.top_k, args.vllm_tp)
    if 5 in args.stages:
        stage5_hcq_computation(args.datasets, data, args.output_dir, args.seeds, args.gpu_id)
    if 51 in args.stages:
        stage5b_enrich_hcq(args.datasets, args.output_dir, args.seeds)
    if 7 in args.stages:
        stage7_threshold_and_ais(args.datasets, data, args.base_output_dir, args.output_dir,
                                  args.seeds, args.dev_size, args.top_k, args.top_k_dual,
                                  EMBEDDINGS_PATH, args.chunk_size, args.gpu_id)
    if 8 in args.stages:
        stage8_vllm_enhanced(args.datasets, data, args.base_output_dir, args.output_dir,
                              args.seeds, args.dev_size, args.top_k, args.vllm_tp)
    if 9 in args.stages:
        stage9_evaluate(args.datasets, data, args.base_output_dir, args.output_dir,
                        args.seeds, args.dev_size, args.top_k, args.results_dir)

    logger.info("All stages complete.")


if __name__ == "__main__":
    main()
