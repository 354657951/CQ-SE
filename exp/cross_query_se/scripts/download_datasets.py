# Download QA benchmark datasets for DTR-aligned experiments.
# Datasets: NQ, WebQuestions, TriviaQA, HotpotQA, SQuAD
# Uses HuggingFace datasets library to download and convert to DTR format.
import os
import json
import sys
import pandas as pd
from datasets import load_dataset

BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)
print(f"Data dir: {BASE_DIR}")
os.makedirs(BASE_DIR, exist_ok=True)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data)} items -> {path}")


def download_nq():
    print("\n=== NaturalQuestions ===")
    out_path = os.path.join(BASE_DIR, "nq/nq-test-contriever.json")
    if os.path.exists(out_path):
        print("  Already exists, skipping.")
        return
    ds = load_dataset("nq_open", split="validation", trust_remote_code=True)
    records = []
    for item in ds:
        records.append({
            "question": item["question"],
            "answers": item["answer"],
        })
    save_json(records, out_path)


def download_webqa():
    print("\n=== WebQuestions ===")
    out_path = os.path.join(BASE_DIR, "webqa/wq-test-contriever.json")
    if os.path.exists(out_path):
        print("  Already exists, skipping.")
        return
    ds = load_dataset("web_questions", split="test", trust_remote_code=True)
    records = []
    for item in ds:
        records.append({
            "question": item["question"],
            "answers": item["answers"],
        })
    save_json(records, out_path)


def download_triviaqa():
    print("\n=== TriviaQA ===")
    out_path = os.path.join(BASE_DIR, "TriviaQA/unfiltered-web-dev.json")
    if os.path.exists(out_path):
        print("  Already exists, skipping.")
        return
    # Use streaming to avoid downloading large train split files
    ds = load_dataset("trivia_qa", "unfiltered", split="validation", streaming=True)
    data_items = []
    for item in ds:
        data_items.append({
            "Question": item["question"],
            "Answer": {
                "Aliases": item["answer"]["aliases"],
                "Value": item["answer"]["value"],
            },
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"Data": data_items}, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data_items)} items -> {out_path}")


def download_hotpotqa():
    print("\n=== HotpotQA ===")
    out_path = os.path.join(BASE_DIR, "hotpotqa/test_qa_pairs.json")
    if os.path.exists(out_path):
        print("  Already exists, skipping.")
        return
    # Use streaming to avoid downloading train split
    ds = load_dataset("hotpot_qa", "fullwiki", split="validation", streaming=True)
    records = []
    for item in ds:
        records.append({
            "question": item["question"],
            "answers": [item["answer"]],
            "id": item["id"],
        })
    save_json(records, out_path)


def download_squad():
    print("\n=== SQuAD ===")
    out_path = os.path.join(BASE_DIR, "SQuAD/validation-00000-of-00001.parquet")
    if os.path.exists(out_path):
        print("  Already exists, skipping.")
        return
    ds = load_dataset("rajpurkar/squad", split="validation", trust_remote_code=True)
    df = ds.to_pandas()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  Saved {len(df)} items -> {out_path}")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["nq", "webqa", "triviaqa", "hotpotqa", "squad"]
    fn_map = {
        "nq": download_nq,
        "webqa": download_webqa,
        "triviaqa": download_triviaqa,
        "hotpotqa": download_hotpotqa,
        "squad": download_squad,
    }
    for t in targets:
        fn_map[t]()
    print("\nAll dataset downloads complete.")
