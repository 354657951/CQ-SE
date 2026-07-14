# INTRYGUE-style induction-aware entropy gating baseline pipeline for adaptive retrieval.
# Supports Qwen2.5-7B-Instruct (single GPU) and Qwen2.5-72B-Instruct (device_map=auto, 8 GPUs).
# Reuses SUGAR baseline outputs (retrieval.jsonl, retrieval_info.jsonl, vllm_pass1.jsonl).
# Stages:
#   1. Induction head calibration (HF eager attention, ~50 NQ examples)
#   2. INTRYGUE scoring via single forward pass (prompt + direct_answer from SUGAR)
#   3. Threshold tuning (dev split, 10 quantile candidates per variant)
#   4. AIS doc selection for enhanced-retrieval queries
#   5. vLLM enhanced answer generation
#   6. Evaluation (EM, F1, AUROC, retrieval cost)

import os
import sys
import json
import gc
import logging
import argparse
import math
import numpy as np
import torch
from collections import defaultdict
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from DTR.dataset.load_data import load_test_qa
from DTR.evaluation.metrics import normalize_answer, exact_match_score, f1_score
from cross_query_se.adaptive._ais import select_topk_of_query_info
from cross_query_se.adaptive.se_trigger import apply_se_trigger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["nq", "webqa", "triviaqa", "hotpotqa", "squad"]
CORPUS_PATH = "data/21MWiki/psgs_w100.tsv"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SUGAR_OUTPUT_DIR = "cross_query_se/outputs/sugar_baseline"


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
    msg = [{"role": "user", "content": f"\n            Question: {q}\n\n            Answer the question using a single word or phrase.\n        "}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def _build_prompt_with_docs(q: str, docs: List[Dict], tokenizer) -> str:
    context = "\n".join(f"Title: {d['title']}. Content: {d['text']}" for d in docs)
    msg = [{"role": "user", "content": f"\n            Question: {q}\n\n            Context: {context}\n\n            Answer the question based on the above context using a single word or phrase.\n        "}]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


# ─── Stage 1: Induction head calibration ────────────────────────────────────

def stage1_calibrate_induction_heads(
    output_dir: str,
    hf_gpu_id: int,
    top_k_heads: int = 10,
    n_calib: int = 50,
    model_name: str = DEFAULT_QWEN_MODEL,
    sugar_output_dir: str = SUGAR_OUTPUT_DIR,
):
    logger.info("=== Stage 1: Induction head calibration ===")
    out_path = os.path.join(output_dir, "induction_heads.json")
    if os.path.exists(out_path):
        logger.info("Stage 1 already done, skipping")
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from cross_query_se.uncertainty.intrygue import identify_induction_heads

    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None

    is_large_model = "72b" in model_name.lower() or "70b" in model_name.lower()
    if is_large_model:
        device_map = "auto"
        device = "cuda:0"
        logger.info(f"Loading {model_name} with eager attention using device_map=auto (multi-GPU)...")
    else:
        device = f"cuda:{hf_gpu_id}"
        device_map = device
        logger.info(f"Loading {model_name} with eager attention on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        dtype=torch.float16,
        attn_implementation="eager",
        device_map=device_map,
    )
    model.eval()

    # Load first n_calib NQ examples as calibration — use short prompt + known answer
    sugar_nq_pass1 = _jsonl_load(os.path.join(sugar_output_dir, "nq_vllm_pass1.jsonl"))
    calib_records = sugar_nq_pass1[:n_calib]

    calib_texts = []
    for rec in calib_records:
        q = rec["question"]
        ans = rec.get("direct_answer", "") or ""
        # Use short prompt+answer for calibration to avoid OOM
        prompt = _build_prompt_direct(q, tokenizer)
        # Concatenate prompt + answer (truncated)
        full = prompt + " " + ans[:50]
        calib_texts.append(full)

    logger.info(f"Running induction head identification on {len(calib_texts)} calibration prompts...")
    top_heads = identify_induction_heads(
        model=model,
        tokenizer=tokenizer,
        calib_texts=calib_texts,
        top_k=top_k_heads,
        device=device,
        max_length=128,
    )

    result = {"top_heads": [[int(l), int(h)] for l, h in top_heads]}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved induction heads to {out_path}: {top_heads}")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Stage 1 done")


# ─── Stage 2: INTRYGUE scoring via single forward pass ──────────────────────

def stage2_intrygue_scoring(
    datasets: List[str],
    data: Dict,
    output_dir: str,
    sugar_output_dir: str,
    hf_gpu_id: int,
    model_name: str = DEFAULT_QWEN_MODEL,
):
    logger.info("=== Stage 2: INTRYGUE scoring (single forward pass per query) ===")

    heads_path = os.path.join(output_dir, "induction_heads.json")
    with open(heads_path) as f:
        heads_data = json.load(f)
    induction_heads = [(l, h) for l, h in heads_data["top_heads"]]
    logger.info(f"Using induction heads: {induction_heads}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from cross_query_se.uncertainty.intrygue import INTRYGUEScorer

    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None

    is_large_model = "72b" in model_name.lower() or "70b" in model_name.lower()
    if is_large_model:
        device_map = "auto"
        device = "cuda:0"
        logger.info(f"Loading {model_name} with eager attention using device_map=auto (multi-GPU)...")
    else:
        device = f"cuda:{hf_gpu_id}"
        device_map = device
        logger.info(f"Loading {model_name} with eager attention on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        dtype=torch.float16,
        attn_implementation="eager",
        device_map=device_map,
    )
    model.eval()

    scorer = INTRYGUEScorer(
        model=model,
        tokenizer=tokenizer,
        induction_heads=induction_heads,
        device=device,
    )

    for ds in datasets:
        out_path = os.path.join(output_dir, f"{ds}_intrygue_scores.jsonl")
        if os.path.exists(out_path):
            logger.info(f"[{ds}] Stage 2 already done, skipping")
            continue

        # Load SUGAR pass1 for pre-generated direct answers
        # Limit to the same questions as data[ds] (respects --num_samples)
        allowed_questions = {ex["question"] for ex in data[ds]}
        sugar_pass1_all = _jsonl_load(os.path.join(sugar_output_dir, f"{ds}_vllm_pass1.jsonl"))
        sugar_pass1 = [r for r in sugar_pass1_all if r["question"] in allowed_questions]
        n = len(sugar_pass1)
        logger.info(f"[{ds}] Computing INTRYGUE scores for {n} queries (filtered from {len(sugar_pass1_all)})...")

        records = []
        for i, rec in enumerate(sugar_pass1):
            if i % 500 == 0:
                logger.info(f"[{ds}] {i}/{n}")
            q = rec["question"]
            direct_answer = rec.get("direct_answer", "") or ""
            prompt = _build_prompt_direct(q, tokenizer)
            try:
                scores = scorer.score_from_answer(prompt, direct_answer)
            except Exception as e:
                logger.warning(f"[{ds}] Error at {i}: {e}")
                scores = {"min_max": 0.0, "mean": 0.0}

            records.append({
                "question": q,
                "answers": rec["answers"],
                "direct_answer": direct_answer,
                "rag3_answer": rec.get("rag3_answer", ""),
                "intrygue_minmax": scores["min_max"],
                "intrygue_mean": scores["mean"],
            })

            if (i + 1) % 2000 == 0:
                _jsonl_save(records, out_path)
                logger.info(f"[{ds}] Checkpoint saved at {i+1}/{n}")

        _jsonl_save(records, out_path)
        logger.info(f"[{ds}] Saved {len(records)} INTRYGUE score records to {out_path}")

    del model, scorer
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Stage 2 done")


# ─── Stage 3: Threshold tuning ──────────────────────────────────────────────

def stage3_threshold_tuning(
    datasets: List[str],
    output_dir: str,
    dev_size: int,
    n_quantiles: int = 10,
):
    logger.info("=== Stage 3: Threshold tuning ===")
    thresholds_path = os.path.join(output_dir, "intrygue_thresholds.json")

    thresholds = {}
    if os.path.exists(thresholds_path):
        with open(thresholds_path) as f:
            thresholds = json.load(f)

    for ds in datasets:
        if ds in thresholds:
            logger.info(f"[{ds}] Thresholds already tuned, skipping")
            continue

        logger.info(f"[{ds}] Tuning thresholds on dev split [0:{dev_size}]...")
        records = _jsonl_load(os.path.join(output_dir, f"{ds}_intrygue_scores.jsonl"))
        dev_records = records[:dev_size]

        best = {}
        for variant, score_key in [("min_max", "intrygue_minmax"), ("mean", "intrygue_mean")]:
            scores_arr = np.array([r[score_key] for r in dev_records])
            q_low = np.linspace(0.05, 0.5, n_quantiles)
            q_high = np.linspace(0.5, 0.95, n_quantiles)
            tau_low_cands = [float(np.quantile(scores_arr, q)) for q in q_low]
            tau_high_cands = [float(np.quantile(scores_arr, q)) for q in q_high]

            best_tau_low = tau_low_cands[0]
            best_tau_high = tau_high_cands[-1]
            best_em = -1.0

            for tau_low in tau_low_cands:
                for tau_high in tau_high_cands:
                    if tau_low >= tau_high:
                        continue
                    total_em = 0.0
                    for rec in dev_records:
                        score = rec[score_key]
                        answers = rec["answers"]
                        decision = apply_se_trigger(score, tau_low, tau_high)
                        if decision == "no_retrieval":
                            pred = rec["direct_answer"]
                        else:
                            pred = rec.get("rag3_answer", rec["direct_answer"])
                        total_em += _em(pred, answers)
                    avg_em = total_em / len(dev_records) if dev_records else 0.0
                    if avg_em > best_em:
                        best_em = avg_em
                        best_tau_low = tau_low
                        best_tau_high = tau_high

            best[variant] = {
                "tau_low": float(best_tau_low),
                "tau_high": float(best_tau_high),
                "dev_em": float(best_em),
            }
            logger.info(
                f"[{ds}] variant={variant} tau_low={best_tau_low:.6f}, "
                f"tau_high={best_tau_high:.6f}, dev_EM={best_em:.4f}"
            )

        thresholds[ds] = best
        with open(thresholds_path, "w") as f:
            json.dump(thresholds, f, indent=2)

    logger.info("Stage 3 done")
    return thresholds


# ─── Stage 4: AIS doc selection ──────────────────────────────────────────────

def stage4_ais_selection(
    datasets: List[str],
    output_dir: str,
    sugar_output_dir: str,
    thresholds: Dict,
    seeds: List[int],
    dev_size: int,
    top_k: int,
):
    logger.info("=== Stage 4: AIS doc selection ===")

    for ds in datasets:
        ret_query = {r["question"]: r for r in _jsonl_load(
            os.path.join(sugar_output_dir, f"{ds}_retrieval.jsonl")
        )}
        ret_info = {r["question"]: r for r in _jsonl_load(
            os.path.join(sugar_output_dir, f"{ds}_retrieval_info.jsonl")
        )}
        records = _jsonl_load(os.path.join(output_dir, f"{ds}_intrygue_scores.jsonl"))
        test_records = records[dev_size:]

        for variant in ["min_max", "mean"]:
            tau_low = thresholds[ds][variant]["tau_low"]
            tau_high = thresholds[ds][variant]["tau_high"]
            score_key = "intrygue_minmax" if variant == "min_max" else "intrygue_mean"

            for seed in seeds:
                out_path = os.path.join(output_dir, f"{ds}_{variant}_enhanced_docs_seed{seed}.jsonl")
                if os.path.exists(out_path):
                    logger.info(f"[{ds}] {variant} seed={seed} AIS already done, skipping")
                    continue

                enhanced_qs = []
                for rec in test_records:
                    score = rec[score_key]
                    if apply_se_trigger(score, tau_low, tau_high) == "enhanced_retrieval":
                        enhanced_qs.append(rec["question"])

                logger.info(f"[{ds}] {variant} seed={seed}: {len(enhanced_qs)} enhanced-retrieval queries")

                enhanced_docs_map = {}
                if enhanced_qs:
                    batch_doc_ids_q, batch_doc_embs_q, batch_D_q, batch_query_embs = [], [], [], []
                    batch_doc_ids_info, batch_doc_embs_info, batch_D_info, batch_info_embs = [], [], [], []
                    valid_qs = []
                    for q in enhanced_qs:
                        rq = ret_query.get(q)
                        ri = ret_info.get(q)
                        if rq is None or ri is None:
                            continue
                        batch_doc_ids_q.append(rq["doc_ids"])
                        batch_doc_embs_q.append(rq["doc_embs"])
                        batch_D_q.append(rq["scores"])
                        batch_query_embs.append(rq["query_emb"])
                        batch_doc_ids_info.append(ri["doc_ids_info"])
                        batch_doc_embs_info.append(ri["doc_embs_info"])
                        batch_D_info.append(ri["scores_info"])
                        batch_info_embs.append(ri["info_emb"])
                        valid_qs.append(q)

                    if valid_qs:
                        selected_doc_ids, _ = select_topk_of_query_info(
                            np.array(batch_doc_ids_q), np.array(batch_doc_ids_info),
                            np.array(batch_doc_embs_q), np.array(batch_doc_embs_info),
                            np.array(batch_D_q), np.array(batch_D_info),
                            np.array(batch_query_embs), np.array(batch_info_embs),
                            topk_new=top_k,
                        )
                        for j, q in enumerate(valid_qs):
                            enhanced_docs_map[q] = [int(x) for x in selected_doc_ids[j]]

                out_records = []
                for rec in test_records:
                    q = rec["question"]
                    score = rec[score_key]
                    decision = apply_se_trigger(score, tau_low, tau_high)
                    if decision == "enhanced_retrieval" and q in enhanced_docs_map:
                        out_records.append({"question": q, "decision": "enhanced_retrieval",
                                           "doc_ids": enhanced_docs_map[q]})
                    else:
                        out_records.append({"question": q, "decision": decision, "doc_ids": []})
                _jsonl_save(out_records, out_path)
                logger.info(f"[{ds}] {variant} seed={seed}: Saved {len(out_records)} AIS records")

    logger.info("Stage 4 done")


# ─── Stage 5: vLLM enhanced answer generation ────────────────────────────────

def stage5_vllm_enhanced(
    datasets: List[str],
    output_dir: str,
    sugar_output_dir: str,
    seeds: List[int],
    dev_size: int,
    top_k: int,
    vllm_tp: int,
    model_name: str = DEFAULT_QWEN_MODEL,
):
    logger.info("=== Stage 5: vLLM enhanced answer generation ===")

    needs_vllm = any(
        not os.path.exists(os.path.join(output_dir, f"{ds}_{variant}_enhanced_answers_seed{seed}.jsonl"))
        for ds in datasets for variant in ["min_max", "mean"] for seed in seeds
    )
    if not needs_vllm:
        logger.info("All enhanced answers already done, skipping Stage 5")
        return

    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
    from transformers import AutoTokenizer

    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=vllm_tp,
        gpu_memory_utilization=0.85,
        max_num_seqs=32,
        max_model_len=4096,
        download_dir=cache_dir,
    )
    greedy_params = SamplingParams(n=1, temperature=0, top_p=1.0, max_tokens=64, stop=["<|im_end|>"])

    for ds in datasets:
        # Collect all doc IDs needed
        all_doc_ids = set()
        for variant in ["min_max", "mean"]:
            for seed in seeds:
                ais_path = os.path.join(output_dir, f"{ds}_{variant}_enhanced_docs_seed{seed}.jsonl")
                if os.path.exists(ais_path):
                    for r in _jsonl_load(ais_path):
                        if r["decision"] == "enhanced_retrieval":
                            all_doc_ids.update(r.get("doc_ids", []))
        corpus_texts = _load_corpus_texts(all_doc_ids, CORPUS_PATH) if all_doc_ids else {}

        for variant in ["min_max", "mean"]:
            for seed in seeds:
                out_path = os.path.join(output_dir, f"{ds}_{variant}_enhanced_answers_seed{seed}.jsonl")
                if os.path.exists(out_path):
                    logger.info(f"[{ds}] {variant} seed={seed} Enhanced answers already done, skipping")
                    continue

                ais_path = os.path.join(output_dir, f"{ds}_{variant}_enhanced_docs_seed{seed}.jsonl")
                ais_records = _jsonl_load(ais_path)
                enhanced_qs = [r["question"] for r in ais_records if r["decision"] == "enhanced_retrieval"]
                enhanced_doc_ids = {r["question"]: r["doc_ids"] for r in ais_records if r["decision"] == "enhanced_retrieval"}

                logger.info(f"[{ds}] {variant} seed={seed}: Generating enhanced answers for {len(enhanced_qs)} queries...")

                if enhanced_qs:
                    prompts = []
                    for q in enhanced_qs:
                        doc_ids = enhanced_doc_ids.get(q, [])
                        docs = [corpus_texts[did] for did in doc_ids if did in corpus_texts]
                        prompts.append(_build_prompt_with_docs(q, docs, tokenizer))
                    outputs = llm.generate(prompts, greedy_params)
                    answers = [o.outputs[0].text.strip() for o in outputs]
                else:
                    answers = []

                records = [{"question": q, "enhanced_answer": ans}
                           for q, ans in zip(enhanced_qs, answers)]
                _jsonl_save(records, out_path)
                logger.info(f"[{ds}] {variant} seed={seed}: Saved {len(records)} enhanced answers")

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Stage 5 done")


# ─── Stage 6: Evaluation ────────────────────────────────────────────────────

def stage6_evaluate(
    datasets: List[str],
    output_dir: str,
    thresholds: Dict,
    seeds: List[int],
    dev_size: int,
    top_k: int,
    results_dir: str,
):
    logger.info("=== Stage 6: Evaluation ===")
    from sklearn.metrics import roc_auc_score

    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    for ds in datasets:
        score_records = _jsonl_load(os.path.join(output_dir, f"{ds}_intrygue_scores.jsonl"))
        test_records = score_records[dev_size:]

        variant_results = {}
        for variant in ["min_max", "mean"]:
            tau_low = thresholds[ds][variant]["tau_low"]
            tau_high = thresholds[ds][variant]["tau_high"]
            score_key = "intrygue_minmax" if variant == "min_max" else "intrygue_mean"

            seed_metrics = []
            for seed in seeds:
                ais_by_q = {}
                ais_path = os.path.join(output_dir, f"{ds}_{variant}_enhanced_docs_seed{seed}.jsonl")
                if os.path.exists(ais_path):
                    for r in _jsonl_load(ais_path):
                        ais_by_q[r["question"]] = r

                enh_by_q = {}
                enh_path = os.path.join(output_dir, f"{ds}_{variant}_enhanced_answers_seed{seed}.jsonl")
                if os.path.exists(enh_path):
                    for r in _jsonl_load(enh_path):
                        enh_by_q[r["question"]] = r["enhanced_answer"]

                em_list, f1_list, score_list, correct_list = [], [], [], []
                n_no_ret = n_single_ret = n_enhanced_ret = 0
                total_retriever_calls = 0

                for rec in test_records:
                    q = rec["question"]
                    answers = rec["answers"]
                    score = rec[score_key]
                    decision = apply_se_trigger(score, tau_low, tau_high)

                    if decision == "no_retrieval":
                        pred = rec["direct_answer"]
                        n_no_ret += 1
                    elif decision == "single_retrieval":
                        pred = rec.get("rag3_answer", rec["direct_answer"])
                        n_single_ret += 1
                        total_retriever_calls += 1
                    else:
                        pred = enh_by_q.get(q, rec.get("rag3_answer", rec["direct_answer"]))
                        n_enhanced_ret += 1
                        total_retriever_calls += 2

                    em = _em(pred, answers)
                    f1 = _f1(pred, answers)
                    em_list.append(em)
                    f1_list.append(f1)
                    score_list.append(score)
                    correct_list.append(int(em > 0))

                n_test = len(test_records)
                avg_em = float(np.mean(em_list)) if em_list else 0.0
                avg_f1 = float(np.mean(f1_list)) if f1_list else 0.0
                avg_ret_calls = total_retriever_calls / max(n_test, 1)
                try:
                    auroc = float(roc_auc_score(correct_list, score_list))
                except Exception:
                    auroc = float("nan")

                seed_metrics.append({
                    "seed": seed,
                    "em": avg_em,
                    "f1": avg_f1,
                    "auroc": auroc,
                    "avg_retriever_calls": avg_ret_calls,
                    "n_test": n_test,
                    "n_no_retrieval": n_no_ret,
                    "n_single_retrieval": n_single_ret,
                    "n_enhanced_retrieval": n_enhanced_ret,
                })
                logger.info(
                    f"[{ds}] {variant} seed={seed} EM={avg_em:.4f} F1={avg_f1:.4f} "
                    f"AUROC={auroc:.4f} RetCalls={avg_ret_calls:.2f}"
                )

            em_vals = [m["em"] for m in seed_metrics]
            f1_vals = [m["f1"] for m in seed_metrics]
            auroc_vals = [m["auroc"] for m in seed_metrics if not math.isnan(m["auroc"])]
            rc_vals = [m["avg_retriever_calls"] for m in seed_metrics]

            variant_results[variant] = {
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
                "per_seed": seed_metrics,
            }
            logger.info(
                f"[{ds}] {variant} FINAL EM={variant_results[variant]['em_mean']:.4f}±{variant_results[variant]['em_std']:.4f} "
                f"AUROC={variant_results[variant]['auroc_mean']:.4f}±{variant_results[variant]['auroc_std']:.4f}"
            )

        best_variant = max(["min_max", "mean"], key=lambda v: thresholds[ds][v].get("dev_em", 0.0))
        all_results[ds] = {
            "best_variant": best_variant,
            "variants": variant_results,
            "em_mean": variant_results[best_variant]["em_mean"],
            "em_std": variant_results[best_variant]["em_std"],
            "f1_mean": variant_results[best_variant]["f1_mean"],
            "f1_std": variant_results[best_variant]["f1_std"],
            "auroc_mean": variant_results[best_variant]["auroc_mean"],
            "auroc_std": variant_results[best_variant]["auroc_std"],
            "avg_retriever_calls_mean": variant_results[best_variant]["avg_retriever_calls_mean"],
            "n_test": variant_results[best_variant]["n_test"],
        }

    summary_path = os.path.join(results_dir, "intrygue_baseline_results.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {summary_path}")

    print("\n=== INTRYGUE BASELINE RESULTS ===")
    print(f"{'Dataset':<12} {'Variant':<12} {'EM':>10} {'F1':>10} {'AUROC':>10} {'RetCalls':>10}")
    print("-" * 65)
    for ds, r in all_results.items():
        for v in ["min_max", "mean"]:
            vr = r["variants"][v]
            marker = " *" if v == r["best_variant"] else "  "
            print(
                f"{ds:<12} {(v + marker):<12} {vr['em_mean']:>7.4f}±{vr['em_std']:.3f} "
                f"{vr['f1_mean']:>7.4f}±{vr['f1_std']:.3f} "
                f"{vr['auroc_mean']:>7.4f}±{vr['auroc_std']:.3f} "
                f"{vr['avg_retriever_calls_mean']:>8.2f}"
            )

    return all_results


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL, help="HuggingFace model name")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--output_dir", default="cross_query_se/outputs/intrygue_baseline")
    parser.add_argument("--results_dir", default="cross_query_se/results/intrygue_baseline")
    parser.add_argument("--sugar_output_dir", default=SUGAR_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dev_size", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--hf_gpu_id", type=int, default=7)
    parser.add_argument("--vllm_tp", type=int, default=4)
    parser.add_argument("--top_k_heads", type=int, default=10)
    parser.add_argument("--n_calib", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--stages", nargs="+", type=int, default=list(range(1, 7)))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets (for metadata only; actual answers come from sugar_output_dir)
    data = {}
    for ds in args.datasets:
        examples = load_test_qa(ds, num_samples=args.num_samples)
        data[ds] = examples
        logger.info(f"Loaded {len(examples)} examples for {ds}")

    if 1 in args.stages:
        stage1_calibrate_induction_heads(
            output_dir=args.output_dir,
            hf_gpu_id=args.hf_gpu_id,
            top_k_heads=args.top_k_heads,
            n_calib=args.n_calib,
            model_name=args.model,
            sugar_output_dir=args.sugar_output_dir,
        )

    if 2 in args.stages:
        stage2_intrygue_scoring(
            datasets=args.datasets,
            data=data,
            output_dir=args.output_dir,
            sugar_output_dir=args.sugar_output_dir,
            hf_gpu_id=args.hf_gpu_id,
            model_name=args.model,
        )

    thresholds = {}
    th_path = os.path.join(args.output_dir, "intrygue_thresholds.json")

    if 3 in args.stages:
        thresholds = stage3_threshold_tuning(
            datasets=args.datasets,
            output_dir=args.output_dir,
            dev_size=args.dev_size,
            n_quantiles=10,
        )
    elif os.path.exists(th_path):
        with open(th_path) as f:
            thresholds = json.load(f)

    if 4 in args.stages:
        stage4_ais_selection(
            datasets=args.datasets,
            output_dir=args.output_dir,
            sugar_output_dir=args.sugar_output_dir,
            thresholds=thresholds,
            seeds=args.seeds,
            dev_size=args.dev_size,
            top_k=args.top_k,
        )

    if 5 in args.stages:
        stage5_vllm_enhanced(
            datasets=args.datasets,
            output_dir=args.output_dir,
            sugar_output_dir=args.sugar_output_dir,
            seeds=args.seeds,
            dev_size=args.dev_size,
            top_k=args.top_k,
            vllm_tp=args.vllm_tp,
            model_name=args.model,
        )

    if 6 in args.stages:
        stage6_evaluate(
            datasets=args.datasets,
            output_dir=args.output_dir,
            thresholds=thresholds,
            seeds=args.seeds,
            dev_size=args.dev_size,
            top_k=args.top_k,
            results_dir=args.results_dir,
        )

    logger.info("All stages complete.")


if __name__ == "__main__":
    main()
