# Main feasibility gate experiment script.
# For each of 5 QA datasets: load 500 examples, load saved perturbations, apply semantic filter,
# then batch-retrieve top-3 doc indices for all examples in a single corpus pass,
# compute V_ret (Jaccard distance on doc index sets), save results.

import os
import sys
import json
import logging
import argparse
import random
from typing import List, Dict, Set

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DTR.dataset.load_data import load_test_qa
from cross_query_se.perturbation.filter import SemanticEquivalenceFilter
from cross_query_se.perturbation.generator import PerturbationGenerator
from cross_query_se.retrieval.bge_retriever import ChunkedBGERetriever
from cross_query_se.analysis.retrieval_variance import (
    compute_v_ret,
    compute_dataset_stats,
    plot_v_ret_histograms,
    apply_feasibility_gate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["nq", "webqa", "triviaqa", "hotpotqa", "squad"]
NUM_SAMPLES = 500
K = 5
TOP_K = 3
SEED = 42
EMBEDDINGS_PATH = "data/21MWiki_bge/corpus_embeddings.npy"
BGE_MODEL = "BAAI/bge-large-en-v1.5"


def load_perturbations(dataset: str, pert_dir: str) -> Dict[str, List[str]]:
    path = os.path.join(pert_dir, f"{dataset}_perturbations.jsonl")
    if not os.path.exists(path):
        return {}
    cache = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cache[obj["question"]] = obj["perturbations"]
    logger.info(f"Loaded {len(cache)} cached perturbations for {dataset}")
    return cache


def run_dataset(
    data_name: str,
    filt: SemanticEquivalenceFilter,
    generator: PerturbationGenerator,
    pert_cache: Dict[str, List[str]],
    retriever: ChunkedBGERetriever,
    output_dir: str,
    k: int = K,
    top_k: int = TOP_K,
    num_samples: int = NUM_SAMPLES,
) -> List[float]:
    logger.info(f"=== Dataset: {data_name} ===")
    random.seed(SEED)
    np.random.seed(SEED)

    out_path = os.path.join(output_dir, f"{data_name}_results.jsonl")
    existing_questions = set()
    existing_records = {}
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing_questions.add(obj["question"])
                    existing_records[obj["question"]] = obj
                except Exception:
                    pass
        logger.info(f"[{data_name}] Resuming: {len(existing_questions)} already done")

    data = load_test_qa(data_name, num_samples=num_samples)
    to_process = [ex for ex in data if ex["question"] not in existing_questions]
    logger.info(f"[{data_name}] {len(to_process)} examples to process")

    # Step 1: Filter all perturbations (GPU: NLI + embedding)
    logger.info(f"[{data_name}] Running semantic filter...")
    all_valid_perts = []
    for ex in tqdm(to_process, desc=f"{data_name} filter"):
        question = ex["question"]
        raw_perts = pert_cache.get(question, [])
        if not raw_perts:
            raw_perts = generator.generate(question)
        valid_perts = filt.filter_with_regeneration(
            original=question,
            perturbations=raw_perts,
            generator=generator,
            k=k,
        )
        all_valid_perts.append(valid_perts)

    # Step 2: Batch retrieval — all examples in one corpus pass
    logger.info(f"[{data_name}] Running batch retrieval...")
    all_query_lists = [
        [ex["question"]] + perts
        for ex, perts in zip(to_process, all_valid_perts)
    ]
    all_doc_id_sets = retriever.retrieve_top_k_batch(all_query_lists, k=top_k)

    # Step 3: Compute V_ret and save
    os.makedirs(output_dir, exist_ok=True)
    v_ret_values = [r["v_ret"] for r in existing_records.values()]

    with open(out_path, "a", encoding="utf-8") as f_out:
        for ex, valid_perts, doc_id_sets in zip(to_process, all_valid_perts, all_doc_id_sets):
            v = compute_v_ret(doc_id_sets)
            v_ret_values.append(v)
            record = {
                "question": ex["question"],
                "perturbations": valid_perts,
                "n_valid_perts": len(valid_perts),
                "doc_id_sets": [list(s) for s in doc_id_sets],
                "v_ret": v,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"[{data_name}] Done. Median V_ret={np.median(v_ret_values):.4f}")
    return v_ret_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--pert_dir", default="cross_query_se/outputs/perturbations")
    parser.add_argument("--output_dir", default="cross_query_se/outputs/feasibility_gate")
    parser.add_argument("--results_dir", default="cross_query_se/results/feasibility_gate")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--cosine_only", action="store_true", default=False,
                        help="Use cosine-similarity-only filter (skip NLI). Better for question paraphrases.")
    parser.add_argument("--embeddings_path", type=str, default=EMBEDDINGS_PATH)
    parser.add_argument("--chunk_size", type=int, default=3_000_000)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"

    retriever = ChunkedBGERetriever(
        embeddings_path=args.embeddings_path,
        model_name=BGE_MODEL,
        device=device,
        chunk_size=args.chunk_size,
    )
    filt = SemanticEquivalenceFilter(tau=args.tau, device=device, cosine_only=args.cosine_only)
    generator = PerturbationGenerator(k=args.k)

    all_v_rets = {}
    all_stats = {}

    for ds in args.datasets:
        pert_cache = load_perturbations(ds, args.pert_dir)
        v_ret_values = run_dataset(
            data_name=ds,
            filt=filt,
            generator=generator,
            pert_cache=pert_cache,
            retriever=retriever,
            output_dir=args.output_dir,
            k=args.k,
            top_k=args.top_k,
            num_samples=args.num_samples,
        )
        all_v_rets[ds] = v_ret_values
        stats = compute_dataset_stats(v_ret_values)
        all_stats[ds] = stats
        logger.info(f"[{ds}] mean={stats['mean']:.4f} median={stats['median']:.4f} std={stats['std']:.4f}")

    os.makedirs(args.results_dir, exist_ok=True)
    plot_path = os.path.join(args.results_dir, "v_ret_histograms.png")
    plot_v_ret_histograms(all_v_rets, plot_path)

    gate_result = apply_feasibility_gate(all_stats)
    logger.info(f"Gate result: {gate_result}")

    summary = {
        "dataset_stats": all_stats,
        "gate_result": gate_result,
        "config": {
            "k": args.k,
            "top_k": args.top_k,
            "tau": args.tau,
            "cosine_only": args.cosine_only,
            "num_samples": args.num_samples,
            "embeddings_path": args.embeddings_path,
            "chunk_size": args.chunk_size,
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "feasibility_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved to {summary_path}")

    print("\n=== FEASIBILITY GATE RESULTS ===")
    print(f"{'Dataset':<12} {'Mean':>8} {'Median':>8} {'Std':>8} {'>=0.05':>8}")
    print("-" * 50)
    for ds, stats in all_stats.items():
        print(f"{ds:<12} {stats['mean']:>8.4f} {stats['median']:>8.4f} {stats['std']:>8.4f} {stats['frac_above_005']:>8.3f}")
    print(f"\nGate: {gate_result['verdict']} ({gate_result['n_passing']}/{len(args.datasets)} datasets pass)")


if __name__ == "__main__":
    main()
