# Build BGE FAISS indexes for 21MWiki (NQ/WebQA/TriviaQA/SQuAD) and HotpotQA corpora.
# Downloads 21MWiki corpus if not present, then embeds+indexes with bge-large-en-v1.5.
# Memory-efficient: processes corpus in streaming chunks, saves partial embeddings to disk.

import os
import sys
import argparse
import logging
import subprocess

import numpy as np
import faiss
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WIKI_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
WIKI_DIR = "data/21MWiki"
WIKI_TSV = "data/21MWiki/psgs_w100.tsv"
WIKI_INDEX_DIR = "data/21MWiki_bge"

HOTPOT_CORPUS = "data/hotpotqa/corpus.json"
HOTPOT_INDEX_DIR = "data/hotpotqa_bge"

BGE_LARGE = "BAAI/bge-large-en-v1.5"


def download_wiki_corpus():
    os.makedirs(WIKI_DIR, exist_ok=True)
    gz_path = WIKI_TSV + ".gz"
    if os.path.exists(WIKI_TSV):
        logger.info(f"21MWiki corpus already exists at {WIKI_TSV}")
        return
    if not os.path.exists(gz_path):
        logger.info(f"Downloading 21MWiki corpus from {WIKI_URL}")
        subprocess.run(["wget", "-q", "-O", gz_path, WIKI_URL], check=True)
    logger.info(f"Decompressing {gz_path}")
    subprocess.run(["gunzip", "-k", gz_path], check=True)
    logger.info(f"21MWiki corpus ready at {WIKI_TSV}")


def count_tsv_lines(path):
    result = subprocess.run(["wc", "-l", path], capture_output=True, text=True)
    return int(result.stdout.strip().split()[0])


def embed_wiki_streaming(model_name, tsv_path, output_dir, batch_size=512, num_gpus=None):
    """Stream TSV row-by-row, embed in batches, save partial .npy files, then concatenate."""
    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, "corpus_embeddings.npy")

    if os.path.exists(emb_path):
        logger.info(f"Embeddings already exist at {emb_path}, skipping embedding.")
        return

    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No GPUs available"

    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None

    # Use GPU 0 for embedding (or all GPUs via DataParallel-like approach via device_map)
    # Here we use GPU 0 with a large batch — bge-large processes ~5k/s on single GPU
    device = "cuda:0"
    logger.info(f"Loading model {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)

    # Stream TSV and embed
    import csv
    chunk_size = 200000  # 200k passages per chunk file
    chunk_idx = 0
    texts_buf = []
    doc_ids = []

    def flush_chunk(texts, idx):
        if not texts:
            return
        chunk_path = os.path.join(output_dir, f"chunk_{idx:04d}.npy")
        if os.path.exists(chunk_path):
            logger.info(f"Chunk {idx} already exists, skipping.")
            return
        logger.info(f"Embedding chunk {idx}: {len(texts)} texts")
        embs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(chunk_path, embs.astype(np.float32))
        logger.info(f"Saved chunk {idx} to {chunk_path}")

    total_lines = count_tsv_lines(tsv_path)
    logger.info(f"Total lines in TSV (including header): {total_lines}")

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, total=total_lines - 1, desc="Reading TSV"):
            texts_buf.append(str(row.get("text", "")))
            doc_ids.append(str(row.get("id", len(doc_ids))))
            if len(texts_buf) >= chunk_size:
                flush_chunk(texts_buf, chunk_idx)
                chunk_idx += 1
                texts_buf = []

    if texts_buf:
        flush_chunk(texts_buf, chunk_idx)
        chunk_idx += 1

    # Save doc IDs
    ids_path = os.path.join(output_dir, "doc_ids.txt")
    # Re-read to get ordered IDs
    id_list = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, total=total_lines - 1, desc="Re-reading IDs"):
            id_list.append(str(row.get("id", "")))
    with open(ids_path, "w") as f:
        f.write("\n".join(id_list))
    logger.info(f"Saved {len(id_list)} doc IDs to {ids_path}")

    # Concatenate all chunks
    logger.info("Concatenating all chunk embeddings...")
    chunk_files = sorted([
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if f.startswith("chunk_") and f.endswith(".npy")
    ])
    all_embs = [np.load(p) for p in tqdm(chunk_files, desc="Loading chunks")]
    embeddings = np.vstack(all_embs).astype(np.float32)
    logger.info(f"Final embeddings shape: {embeddings.shape}")
    np.save(emb_path, embeddings)
    logger.info(f"Saved full embeddings to {emb_path}")

    # Clean up chunks
    for p in chunk_files:
        os.remove(p)


def embed_hotpot(model_name, corpus_path, output_dir, batch_size=512):
    import json
    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, "corpus_embeddings.npy")

    if os.path.exists(emb_path):
        logger.info(f"HotpotQA embeddings already exist at {emb_path}, skipping.")
        return

    hf_home = os.environ.get("HF_HOME", None)
    cache_dir = os.path.join(hf_home, "hub") if hf_home else None

    logger.info(f"Loading HotpotQA corpus from {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    texts = [" ".join(doc["sentences"]) for doc in corpus]
    ids = [str(doc["id"]) for doc in corpus]
    logger.info(f"HotpotQA corpus: {len(texts)} passages")

    # Save IDs
    ids_path = os.path.join(output_dir, "doc_ids.txt")
    with open(ids_path, "w") as f:
        f.write("\n".join(ids))

    device = "cuda:0"
    logger.info(f"Loading model {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32)
    logger.info(f"HotpotQA embeddings shape: {embeddings.shape}")
    np.save(emb_path, embeddings)
    logger.info(f"Saved HotpotQA embeddings to {emb_path}")


def build_faiss_index(embeddings_path, output_dir):
    index_path = os.path.join(output_dir, "faiss_index_emb")
    if os.path.exists(index_path):
        logger.info(f"FAISS index already exists at {index_path}, skipping.")
        return

    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path).astype(np.float32)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    logger.info("Building FAISS IndexFlatIP index")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    chunk = 50000
    for i in tqdm(range(0, len(embeddings), chunk), desc="Adding to FAISS"):
        index.add(embeddings[i : i + chunk])

    faiss.write_index(index, index_path)
    logger.info(f"Saved FAISS index to {index_path} ({index.ntotal} vectors)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=["21mwiki", "hotpotqa", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    if args.corpus in ("21mwiki", "both"):
        download_wiki_corpus()
        embed_wiki_streaming(BGE_LARGE, WIKI_TSV, WIKI_INDEX_DIR, batch_size=args.batch_size)
        emb_path = os.path.join(WIKI_INDEX_DIR, "corpus_embeddings.npy")
        build_faiss_index(emb_path, WIKI_INDEX_DIR)

    if args.corpus in ("hotpotqa", "both"):
        embed_hotpot(BGE_LARGE, HOTPOT_CORPUS, HOTPOT_INDEX_DIR, batch_size=args.batch_size)
        emb_path = os.path.join(HOTPOT_INDEX_DIR, "corpus_embeddings.npy")
        build_faiss_index(emb_path, HOTPOT_INDEX_DIR)

    logger.info("Index building complete.")


if __name__ == "__main__":
    main()
