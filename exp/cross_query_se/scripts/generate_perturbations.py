# Locally run perturbation generation for all 5 datasets (500 examples each).
# Saves results to cross_query_se/outputs/perturbations/{dataset}_perturbations.jsonl
# Uses parallel API calls via ThreadPoolExecutor — CPU/network bound, no GPU needed.

import os
import sys
import json
import logging
import random
import argparse

import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DTR.dataset.load_data import load_test_qa
from cross_query_se.perturbation.generator import PerturbationGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["nq", "webqa", "triviaqa", "hotpotqa", "squad"]
NUM_SAMPLES = 500
K = 5
SEED = 42


def run(datasets, num_samples, k, output_dir, max_workers):
    os.makedirs(output_dir, exist_ok=True)
    generator = PerturbationGenerator(k=k, max_workers=max_workers)

    for ds in datasets:
        out_path = os.path.join(output_dir, f"{ds}_perturbations.jsonl")

        # Load already generated questions to skip
        existing = set()
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        existing.add(obj["question"])
                    except Exception:
                        pass
            logger.info(f"[{ds}] Found {len(existing)} already generated. Will skip.")

        random.seed(SEED)
        np.random.seed(SEED)
        data = load_test_qa(ds, num_samples=num_samples)
        logger.info(f"[{ds}] Loaded {len(data)} examples")

        to_process = [ex for ex in data if ex["question"] not in existing]
        logger.info(f"[{ds}] Processing {len(to_process)} new examples")

        if not to_process:
            continue

        questions = [ex["question"] for ex in to_process]
        all_perts = generator.generate_batch(questions)

        with open(out_path, "a", encoding="utf-8") as f:
            for ex, perts in tqdm(zip(to_process, all_perts), total=len(to_process), desc=ds):
                record = {
                    "question": ex["question"],
                    "answers": ex.get("answers", []),
                    "perturbations": perts,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"[{ds}] Saved perturbations to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--output_dir", default="cross_query_se/outputs/perturbations")
    parser.add_argument("--max_workers", type=int, default=20)
    args = parser.parse_args()

    run(args.datasets, args.num_samples, args.k, args.output_dir, args.max_workers)


if __name__ == "__main__":
    main()
