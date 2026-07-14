# SUGAR-style within-query SE baseline pipeline for adaptive retrieval on Qwen2.5-7B.
# 7-stage pipeline: BGE retrieval x2, vLLM generation x2, DeBERTa SE, threshold tuning, evaluation.
# Each stage checkpoints to disk; re-running resumes from existing outputs.

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
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from DTR.dataset.load_data import load_test_qa
from DTR.evaluation.metrics import normalize_answer, exact_match_score, f1_score
from cross_query_se.retrieval.bge_retriever import ChunkedBGERetriever
from cross_query_se.uncertainty.within_query_se import WithinQuerySE
from cross_query_se.adaptive.se_trigger import apply_se_trigger, tune_se_thresholds
from cross_query_se.adaptive._ais import select_topk_of_query_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["nq", "webqa", "triviaqa", "hotpotqa", "squad"]
EMBEDDINGS_PATH = "data/21MWiki_bge/corpus_embeddings.npy"
CORPUS_PATH = "data/21MWiki/psgs_w100.tsv"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
DEBERTA_MODEL = "microsoft/deberta-v2-xlarge-mnli"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # default; overridden by --model arg


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
    """Scan psgs_w100.tsv once, return {doc_id: {'title': ..., 'text': ...}} for needed_ids."""
    logger.info(f"Scanning corpus for {len(needed_ids)} doc IDs...")
    result = {}
    import pandas as pd
    chunk_iter = pd.read_csv(corpus_path, sep="\t", chunksize=500_000)
    loaded = 0
    for chunk in chunk_iter:
        for _, row in chunk.iterrows():
            did = int(row["id"]) - 1  # psgs_w100 is 1-indexed
            if did in needed_ids:
                result[did] = {"title": str(row["title"]), "text": str(row["text"])}
                loaded += 1
        if loaded >= len(needed_ids):
            break
    logger.info(f"Loaded {loaded} doc texts")
    return result


def _build_prompt_direct(q: str, tokenizer) -> str:
    msg = [{"role": "user", "content": f"\n            Question: {q}\n\n            Answer the question using a single word or phrase.\n        "}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def _build_prompt_with_docs(q: str, docs: List[Dict], tokenizer) -> str:
    context = "\n".join(f"Title: {d['title']}. Content: {d['text']}" for d in docs)
    msg = [{"role": "user", "content": f"\n            Question: {q}\n\n            Context: {context}\n\n            Answer the question based on the above context using a single word or phrase.\n        "}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def _build_prompt_pseudo(q: str, tokenizer) -> str:
    msg = [{"role": "user", "content": f"\n            Please write a passage to answer the question\n\n            Question: {q}\n\n            Passage:\n        "}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


# ─── Stage 1: BGE retrieval (query → top-5 docs + embs) ────────────────────

def stage1_retrieval(datasets, data, output_dir, embeddings_path, chunk_size, top_k_dual, gpu_id):
    logger.info("=== Stage 1: BGE retrieval (queries → top-k docs + embeddings) ===")
    hf_home = os.environ.get("HF_HOME", None)
    device = f"cuda:{gpu_id}"
    retriever = ChunkedBGERetriever(
        embeddings_path=embeddings_path,
        model_name=BGE_MODEL,
        device=device,
        chunk_size=chunk_size,
    )

    for ds in datasets:
        out_path = os.path.join(output_dir, f"{ds}_retrieval.jsonl")
        if os.path.exists(out_path):
            logger.info(f"[{ds}] Stage 1 already done, skipping")
            continue
        logger.info(f"[{ds}] Retrieving top-{top_k_dual} docs for {len(data[ds])} queries...")
        questions = [ex["question"] for ex in data[ds]]
        doc_ids_np, doc_embs_np, scores_np, query_embs_np = retriever.retrieve_top_k_with_embs(
            questions, k=top_k_dual
        )
        records = []
        for i, ex in enumerate(data[ds]):
            records.append({
                "question": ex["question"],
                "doc_ids": doc_ids_np[i].tolist(),
                "doc_embs": doc_embs_np[i].tolist(),
                "scores": scores_np[i].tolist(),
                "query_emb": query_embs_np[i].tolist(),
            })
        _jsonl_save(records, out_path)
        logger.info(f"[{ds}] Saved {len(records)} retrieval records to {out_path}")

    del retriever
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Stage 1 done")


# ─── Stage 2: vLLM generation pass 1 ───────────────────────────────────────

def stage2_vllm_pass1(datasets, data, output_dir, seeds, m_samples, top_k, top_k_dual, vllm_tp, model=None):
    logger.info("=== Stage 2: vLLM generation pass 1 ===")
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    from transformers import AutoTokenizer

    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    _model = model or QWEN_MODEL

    tokenizer = AutoTokenizer.from_pretrained(_model, cache_dir=cache_dir)
    llm = LLM(
        model=_model,
        tensor_parallel_size=vllm_tp,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        max_model_len=4096,
        download_dir=cache_dir,
    )

    for ds in datasets:
        out_path = os.path.join(output_dir, f"{ds}_vllm_pass1.jsonl")
        if os.path.exists(out_path):
            logger.info(f"[{ds}] Stage 2 already done, skipping")
            continue

        retrieval_records = {
            r["question"]: r
            for r in _jsonl_load(os.path.join(output_dir, f"{ds}_retrieval.jsonl"))
        }
        # Load corpus texts for this dataset
        all_doc_ids = set()
        for r in retrieval_records.values():
            all_doc_ids.update(r["doc_ids"])
        corpus_texts = _load_corpus_texts(all_doc_ids, CORPUS_PATH)

        questions = [ex["question"] for ex in data[ds]]
        answers_list = [ex["answers"] for ex in data[ds]]

        # 1) Greedy direct answer
        logger.info(f"[{ds}] Generating greedy direct answers...")
        direct_prompts = [_build_prompt_direct(q, tokenizer) for q in questions]
        greedy_params = SamplingParams(n=1, temperature=0, top_p=1.0, max_tokens=64, stop=["<|im_end|>"])
        direct_outputs = llm.generate(direct_prompts, greedy_params)
        direct_answers = [o.outputs[0].text.strip() for o in direct_outputs]

        # 2) Greedy RAG-3 answer (top-3 from stage-1 retrieval)
        logger.info(f"[{ds}] Generating greedy RAG-{top_k} answers...")
        rag3_prompts = []
        for q in questions:
            r = retrieval_records[q]
            docs = [corpus_texts[did] for did in r["doc_ids"][:top_k] if did in corpus_texts]
            rag3_prompts.append(_build_prompt_with_docs(q, docs, tokenizer))
        rag3_outputs = llm.generate(rag3_prompts, greedy_params)
        rag3_answers = [o.outputs[0].text.strip() for o in rag3_outputs]

        # 3) Pseudo-context for AIS (enhanced path)
        logger.info(f"[{ds}] Generating pseudo-context passages...")
        pseudo_prompts = [_build_prompt_pseudo(q, tokenizer) for q in questions]
        pseudo_outputs = llm.generate(pseudo_prompts, greedy_params)
        pseudo_passages = [o.outputs[0].text.strip() for o in pseudo_outputs]

        # 4) Sampled answers for SE computation (M=5, per seed)
        seed_sampled: Dict[int, List[List[str]]] = {}
        for seed in seeds:
            logger.info(f"[{ds}] Generating M={m_samples} sampled answers (seed={seed})...")
            rng = random.Random(seed)
            # vLLM doesn't support per-request seeds in all versions; use n=M and varied prompts
            sample_params = SamplingParams(
                n=m_samples, temperature=1.0, top_p=0.9, max_tokens=64, stop=["<|im_end|>"]
            )
            # Use RAG-3 context for sampled answers (matching SUGAR: fixed retrieval)
            sample_outputs = llm.generate(rag3_prompts, sample_params)
            seed_answers = []
            for o in sample_outputs:
                samples = [gen.text.strip() for gen in o.outputs]
                # pad/truncate to exactly m_samples
                while len(samples) < m_samples:
                    samples.append(samples[0] if samples else "")
                seed_answers.append(samples[:m_samples])
            seed_sampled[seed] = seed_answers

        # Save
        records = []
        for i, (q, ans) in enumerate(zip(questions, answers_list)):
            rec = {
                "question": q,
                "answers": ans,
                "direct_answer": direct_answers[i],
                "rag3_answer": rag3_answers[i],
                "pseudo_passage": pseudo_passages[i],
                "sampled_answers_by_seed": {str(s): seed_sampled[s][i] for s in seeds},
            }
            records.append(rec)
        _jsonl_save(records, out_path)
        logger.info(f"[{ds}] Saved {len(records)} vllm_pass1 records")

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Stage 2 done")


# ─── Stage 3: BGE retrieval pass 2 (pseudo-context) ────────────────────────

def stage3_retrieval_info(datasets, data, output_dir, embeddings_path, chunk_size, top_k_dual, gpu_id):
    logger.info("=== Stage 3: BGE retrieval pass 2 (pseudo-context) ===")
    device = f"cuda:{gpu_id}"
    retriever = ChunkedBGERetriever(
        embeddings_path=embeddings_path,
        model_name=BGE_MODEL,
        device=device,
        chunk_size=chunk_size,
    )

    for ds in datasets:
        out_path = os.path.join(output_dir, f"{ds}_retrieval_info.jsonl")
        if os.path.exists(out_path):
            logger.info(f"[{ds}] Stage 3 already done, skipping")
            continue
        pass1_records = _jsonl_load(os.path.join(output_dir, f"{ds}_vllm_pass1.jsonl"))
        pseudo_passages = [r["pseudo_passage"] for r in pass1_records]
        questions = [r["question"] for r in pass1_records]

        logger.info(f"[{ds}] Retrieving top-{top_k_dual} docs for {len(pseudo_passages)} pseudo-passages...")
        doc_ids_np, doc_embs_np, scores_np, info_embs_np = retriever.retrieve_top_k_with_embs(
            pseudo_passages, k=top_k_dual
        )
        records = []
        for i, q in enumerate(questions):
            records.append({
                "question": q,
                "doc_ids_info": doc_ids_np[i].tolist(),
                "doc_embs_info": doc_embs_np[i].tolist(),
                "scores_info": scores_np[i].tolist(),
                "info_emb": info_embs_np[i].tolist(),
            })
        _jsonl_save(records, out_path)
        logger.info(f"[{ds}] Saved {len(records)} info retrieval records")

    del retriever
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Stage 3 done")


# ─── Stage 4: DeBERTa SE computation ────────────────────────────────────────

def stage4_se_computation(datasets, data, output_dir, seeds, gpu_id):
    logger.info("=== Stage 4: DeBERTa SE computation ===")
    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    device = f"cuda:{gpu_id}"

    se_estimator = WithinQuerySE(model_name=DEBERTA_MODEL, device=device, batch_size=64, cache_dir=cache_dir)

    for ds in datasets:
        for seed in seeds:
            out_path = os.path.join(output_dir, f"{ds}_se_seed{seed}.jsonl")
            if os.path.exists(out_path):
                logger.info(f"[{ds}] seed={seed} Stage 4 already done, skipping")
                continue
            pass1_records = _jsonl_load(os.path.join(output_dir, f"{ds}_vllm_pass1.jsonl"))
            all_sampled_answers = [r["sampled_answers_by_seed"][str(seed)] for r in pass1_records]

            logger.info(f"[{ds}] seed={seed} Computing SE for {len(all_sampled_answers)} queries...")
            se_scores = se_estimator.compute_se_batch(all_sampled_answers)

            records = []
            for i, r in enumerate(pass1_records):
                records.append({
                    "question": r["question"],
                    "se_score": se_scores[i],
                    "sampled_answers": all_sampled_answers[i],
                })
            _jsonl_save(records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(records)} SE records")

    del se_estimator
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Stage 4 done")


# ─── Stage 5: Threshold tuning + AIS doc selection ──────────────────────────

def stage5_threshold_and_ais(datasets, data, output_dir, seeds, dev_size, top_k, top_k_dual):
    logger.info("=== Stage 5: Threshold tuning + AIS doc selection ===")
    thresholds_path = os.path.join(output_dir, "thresholds.json")

    # Build index: question → position in data[ds]
    thresholds = {}
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            thresholds = json.load(f)

    for ds in datasets:
        if ds in thresholds:
            logger.info(f"[{ds}] Thresholds already tuned, skipping")
        else:
            logger.info(f"[{ds}] Tuning thresholds on dev split [0:{dev_size}]...")
            pass1_records = _jsonl_load(os.path.join(output_dir, f"{ds}_vllm_pass1.jsonl"))
            dev_pass1 = {r["question"]: r for r in pass1_records[:dev_size]}

            # Average SE across seeds for dev set
            avg_se_by_q = defaultdict(list)
            for seed in seeds:
                se_records = _jsonl_load(os.path.join(output_dir, f"{ds}_se_seed{seed}.jsonl"))
                for r in se_records[:dev_size]:
                    avg_se_by_q[r["question"]].append(r["se_score"])

            dev_records = []
            for q, p1r in dev_pass1.items():
                avg_se = float(np.mean(avg_se_by_q[q])) if avg_se_by_q[q] else 0.0
                dev_records.append({
                    "se_score": avg_se,
                    "answers": p1r["answers"],
                    "direct_answer": p1r["direct_answer"],
                    "rag3_answer": p1r["rag3_answer"],
                    "enhanced_answer": None,  # not available at tune time
                })

            tau_low, tau_high = tune_se_thresholds(dev_records)
            thresholds[ds] = {"tau_low": tau_low, "tau_high": tau_high}
            with open(thresholds_path, "w") as f:
                json.dump(thresholds, f, indent=2)
            logger.info(f"[{ds}] tau_low={tau_low}, tau_high={tau_high}")

    # AIS doc selection for enhanced_retrieval queries (test split only)
    for ds in datasets:
        for seed in seeds:
            out_path = os.path.join(output_dir, f"{ds}_enhanced_docs_seed{seed}.jsonl")
            if os.path.exists(out_path):
                logger.info(f"[{ds}] seed={seed} AIS already done, skipping")
                continue

            tau_low = thresholds[ds]["tau_low"]
            tau_high = thresholds[ds]["tau_high"]

            pass1_records = _jsonl_load(os.path.join(output_dir, f"{ds}_vllm_pass1.jsonl"))
            se_records = _jsonl_load(os.path.join(output_dir, f"{ds}_se_seed{seed}.jsonl"))
            ret_query = {r["question"]: r for r in _jsonl_load(os.path.join(output_dir, f"{ds}_retrieval.jsonl"))}
            ret_info = {r["question"]: r for r in _jsonl_load(os.path.join(output_dir, f"{ds}_retrieval_info.jsonl"))}

            # Work on TEST split only (dev_size onwards)
            test_pass1 = pass1_records[dev_size:]
            test_se = se_records[dev_size:]

            # Collect enhanced-trigger queries
            enhanced_questions = []
            enhanced_indices = []
            for i, (p1r, ser) in enumerate(zip(test_pass1, test_se)):
                assert p1r["question"] == ser["question"]
                decision = apply_se_trigger(ser["se_score"], tau_low, tau_high)
                if decision == "enhanced_retrieval":
                    enhanced_questions.append(p1r["question"])
                    enhanced_indices.append(i)

            logger.info(f"[{ds}] seed={seed} {len(enhanced_questions)} queries need enhanced retrieval")

            enhanced_docs_map = {}
            if enhanced_questions:
                # Build arrays for select_topk_of_query_info
                batch_doc_ids_q = []
                batch_doc_embs_q = []
                batch_D_q = []
                batch_query_embs = []
                batch_doc_ids_info = []
                batch_doc_embs_info = []
                batch_D_info = []
                batch_info_embs = []
                for q in enhanced_questions:
                    rq = ret_query[q]
                    ri = ret_info[q]
                    batch_doc_ids_q.append(rq["doc_ids"])
                    batch_doc_embs_q.append(rq["doc_embs"])
                    batch_D_q.append(rq["scores"])
                    batch_query_embs.append(rq["query_emb"])
                    batch_doc_ids_info.append(ri["doc_ids_info"])
                    batch_doc_embs_info.append(ri["doc_embs_info"])
                    batch_D_info.append(ri["scores_info"])
                    batch_info_embs.append(ri["info_emb"])

                doc_ids_q_np = np.array(batch_doc_ids_q)
                doc_embs_q_np = np.array(batch_doc_embs_q)
                scores_q_np = np.array(batch_D_q)
                query_embs_np = np.array(batch_query_embs)
                doc_ids_info_np = np.array(batch_doc_ids_info)
                doc_embs_info_np = np.array(batch_doc_embs_info)
                scores_info_np = np.array(batch_D_info)
                info_embs_np = np.array(batch_info_embs)

                selected_doc_ids, _ = select_topk_of_query_info(
                    doc_ids_q_np, doc_ids_info_np,
                    doc_embs_q_np, doc_embs_info_np,
                    scores_q_np, scores_info_np,
                    query_embs_np, info_embs_np,
                    topk_new=top_k,
                    consider_adaptive=False,
                )
                for q, sel_ids in zip(enhanced_questions, selected_doc_ids):
                    enhanced_docs_map[q] = [int(i) for i in sel_ids]

            # Build full records (all test queries)
            records = []
            for p1r, ser in zip(test_pass1, test_se):
                q = p1r["question"]
                decision = apply_se_trigger(ser["se_score"], tau_low, tau_high)
                rec = {
                    "question": q,
                    "se_score": ser["se_score"],
                    "decision": decision,
                    "enhanced_doc_ids": enhanced_docs_map.get(q, []),
                }
                records.append(rec)
            _jsonl_save(records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(records)} AIS records")

    logger.info("Stage 5 done")


# ─── Stage 6: vLLM generation pass 2 (enhanced answers) ────────────────────

def stage6_vllm_pass2(datasets, data, output_dir, seeds, dev_size, top_k, vllm_tp, model=None):
    logger.info("=== Stage 6: vLLM generation pass 2 (enhanced answers) ===")

    # Check if any enhanced queries exist
    needs_pass2 = False
    for ds in datasets:
        for seed in seeds:
            p = os.path.join(output_dir, f"{ds}_enhanced_docs_seed{seed}.jsonl")
            out_p = os.path.join(output_dir, f"{ds}_enhanced_answers_seed{seed}.jsonl")
            if os.path.exists(out_p):
                continue
            if os.path.exists(p):
                records = _jsonl_load(p)
                if any(r["decision"] == "enhanced_retrieval" for r in records):
                    needs_pass2 = True
                    break
        if needs_pass2:
            break

    if not needs_pass2:
        logger.info("No enhanced queries; skipping Stage 6")
        return

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    from transformers import AutoTokenizer

    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    _model = model or QWEN_MODEL
    tokenizer = AutoTokenizer.from_pretrained(_model, cache_dir=cache_dir)
    llm = LLM(
        model=_model,
        tensor_parallel_size=vllm_tp,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        max_model_len=4096,
        download_dir=cache_dir,
    )
    greedy_params = SamplingParams(n=1, temperature=0, top_p=1.0, max_tokens=64, stop=["<|im_end|>"])

    for ds in datasets:
        # Collect all unique enhanced doc IDs for corpus load
        all_enhanced_doc_ids = set()
        for seed in seeds:
            ais_path = os.path.join(output_dir, f"{ds}_enhanced_docs_seed{seed}.jsonl")
            for r in _jsonl_load(ais_path):
                all_enhanced_doc_ids.update(r["enhanced_doc_ids"])
        corpus_texts = _load_corpus_texts(all_enhanced_doc_ids, CORPUS_PATH) if all_enhanced_doc_ids else {}

        for seed in seeds:
            out_path = os.path.join(output_dir, f"{ds}_enhanced_answers_seed{seed}.jsonl")
            if os.path.exists(out_path):
                logger.info(f"[{ds}] seed={seed} Stage 6 already done, skipping")
                continue

            ais_records = _jsonl_load(os.path.join(output_dir, f"{ds}_enhanced_docs_seed{seed}.jsonl"))
            enhanced_qs = [r["question"] for r in ais_records if r["decision"] == "enhanced_retrieval"]
            enhanced_doc_ids_map = {r["question"]: r["enhanced_doc_ids"] for r in ais_records}

            if not enhanced_qs:
                logger.info(f"[{ds}] seed={seed} No enhanced queries; skipping")
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

            records = []
            for q, ans in zip(enhanced_qs, answers):
                records.append({"question": q, "enhanced_answer": ans})
            _jsonl_save(records, out_path)
            logger.info(f"[{ds}] seed={seed} Saved {len(records)} enhanced answers")

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Stage 6 done")


# ─── Stage 7: Evaluation ────────────────────────────────────────────────────

def stage7_evaluate(datasets, data, output_dir, seeds, dev_size, top_k, results_dir):
    logger.info("=== Stage 7: Evaluation ===")
    from sklearn.metrics import roc_auc_score

    os.makedirs(results_dir, exist_ok=True)
    thresholds = json.load(open(os.path.join(output_dir, "thresholds.json")))

    all_results = {}
    for ds in datasets:
        tau_low = thresholds[ds]["tau_low"]
        tau_high = thresholds[ds]["tau_high"]

        pass1_records = _jsonl_load(os.path.join(output_dir, f"{ds}_vllm_pass1.jsonl"))
        test_pass1 = pass1_records[dev_size:]
        test_pass1_by_q = {r["question"]: r for r in test_pass1}

        seed_metrics = []
        for seed in seeds:
            se_records = _jsonl_load(os.path.join(output_dir, f"{ds}_se_seed{seed}.jsonl"))
            test_se = se_records[dev_size:]
            test_se_by_q = {r["question"]: r for r in test_se}

            ais_records = _jsonl_load(os.path.join(output_dir, f"{ds}_enhanced_docs_seed{seed}.jsonl"))
            ais_by_q = {r["question"]: r for r in ais_records}

            enh_ans_path = os.path.join(output_dir, f"{ds}_enhanced_answers_seed{seed}.jsonl")
            enh_answers_by_q = {}
            if os.path.exists(enh_ans_path):
                for r in _jsonl_load(enh_ans_path):
                    enh_answers_by_q[r["question"]] = r["enhanced_answer"]

            em_list, f1_list, se_list, correct_list = [], [], [], []
            n_no_ret, n_single_ret, n_enhanced_ret = 0, 0, 0
            total_retriever_calls = 0
            total_docs_retrieved = 0

            for p1r in test_pass1:
                q = p1r["question"]
                answers = p1r["answers"]
                ser = test_se_by_q.get(q, {"se_score": 0.0})
                aisr = ais_by_q.get(q, {"decision": "single_retrieval"})

                se = ser["se_score"]
                decision = aisr["decision"]

                if decision == "no_retrieval":
                    pred = p1r["direct_answer"]
                    n_no_ret += 1
                    total_retriever_calls += 0
                    total_docs_retrieved += 0
                elif decision == "single_retrieval":
                    pred = p1r["rag3_answer"]
                    n_single_ret += 1
                    total_retriever_calls += 1
                    total_docs_retrieved += top_k
                else:
                    pred = enh_answers_by_q.get(q, p1r["rag3_answer"])
                    n_enhanced_ret += 1
                    total_retriever_calls += 2  # query + pseudo-context retrieval
                    total_docs_retrieved += top_k

                em = _em(pred, answers)
                f1 = _f1(pred, answers)
                em_list.append(em)
                f1_list.append(f1)
                se_list.append(se)
                correct_list.append(int(em > 0))

            n_test = len(test_pass1)
            avg_em = float(np.mean(em_list)) if em_list else 0.0
            avg_f1 = float(np.mean(f1_list)) if f1_list else 0.0
            avg_ret_calls = total_retriever_calls / max(n_test, 1)
            avg_docs = total_docs_retrieved / max(n_test, 1)

            try:
                auroc = float(roc_auc_score(correct_list, se_list))
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
            "per_seed": seed_metrics,
        }
        logger.info(
            f"[{ds}] FINAL EM={all_results[ds]['em_mean']:.4f}±{all_results[ds]['em_std']:.4f} "
            f"F1={all_results[ds]['f1_mean']:.4f}±{all_results[ds]['f1_std']:.4f} "
            f"AUROC={all_results[ds]['auroc_mean']:.4f}±{all_results[ds]['auroc_std']:.4f}"
        )

    summary_path = os.path.join(results_dir, "sugar_baseline_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {summary_path}")

    # Print summary table
    print("\n=== SUGAR BASELINE RESULTS ===")
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


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--model", default=QWEN_MODEL, help="HF model ID for vLLM generation")
    parser.add_argument("--output_dir", default="cross_query_se/outputs/sugar_baseline")
    parser.add_argument("--results_dir", default="cross_query_se/results/sugar_baseline")
    parser.add_argument("--m_samples", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dev_size", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_k_dual", type=int, default=5)
    parser.add_argument("--chunk_size", type=int, default=500_000)
    parser.add_argument("--vllm_tp", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU for BGE/DeBERTa")
    parser.add_argument("--num_samples", type=int, default=-1, help="Limit dataset size (-1=all)")
    parser.add_argument("--stages", nargs="+", type=int, default=list(range(1, 8)),
                        help="Which stages to run (1-7)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    data = {}
    for ds in args.datasets:
        examples = load_test_qa(ds, num_samples=args.num_samples)
        data[ds] = examples
        logger.info(f"Loaded {len(examples)} examples for {ds}")

    if 1 in args.stages:
        stage1_retrieval(args.datasets, data, args.output_dir, EMBEDDINGS_PATH,
                         args.chunk_size, args.top_k_dual, args.gpu_id)
    if 2 in args.stages:
        stage2_vllm_pass1(args.datasets, data, args.output_dir, args.seeds,
                          args.m_samples, args.top_k, args.top_k_dual, args.vllm_tp, args.model)
    if 3 in args.stages:
        stage3_retrieval_info(args.datasets, data, args.output_dir, EMBEDDINGS_PATH,
                              args.chunk_size, args.top_k_dual, args.gpu_id)
    if 4 in args.stages:
        stage4_se_computation(args.datasets, data, args.output_dir, args.seeds, args.gpu_id)
    if 5 in args.stages:
        stage5_threshold_and_ais(args.datasets, data, args.output_dir, args.seeds,
                                 args.dev_size, args.top_k, args.top_k_dual)
    if 6 in args.stages:
        stage6_vllm_pass2(args.datasets, data, args.output_dir, args.seeds,
                          args.dev_size, args.top_k, args.vllm_tp, args.model)
    if 7 in args.stages:
        stage7_evaluate(args.datasets, data, args.output_dir, args.seeds,
                        args.dev_size, args.top_k, args.results_dir)

    logger.info("All stages complete.")


if __name__ == "__main__":
    main()
