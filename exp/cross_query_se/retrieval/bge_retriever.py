# GPU-accelerated BGE retriever using chunked matrix multiplication.
# Processes ALL queries for a dataset in a single pass over the corpus embeddings.
# Memory: chunk_size * dim * 4 bytes GPU (3M * 1024 * 4 = 12GB per chunk).

import os
import logging
from typing import List, Set

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ChunkedBGERetriever:
    def __init__(
        self,
        embeddings_path: str,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cuda:0",
        chunk_size: int = 3_000_000,
    ):
        self.embeddings_path = embeddings_path
        self.device = device
        self.chunk_size = chunk_size

        hf_home = os.environ.get("HF_HOME", None)
        cache_dir = os.path.join(hf_home, "hub") if hf_home else None

        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading BGE model {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device, cache_folder=cache_dir)

        logger.info(f"Memory-mapping corpus embeddings from {embeddings_path}")
        self.corpus_embs = np.load(embeddings_path, mmap_mode="r")
        self.n_docs = self.corpus_embs.shape[0]
        self.dim = self.corpus_embs.shape[1]
        logger.info(f"Corpus: {self.n_docs} docs, dim={self.dim}")

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        embs = self.model.encode(
            queries,
            device=self.device,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return torch.from_numpy(embs.astype(np.float32)).to(self.device)

    def retrieve_top_k(self, queries: List[str], k: int = 3) -> List[Set[int]]:
        """Retrieve top-k doc indices for a single example's queries (original + perturbations)."""
        query_embs = self.encode_queries(queries)  # [Q, D]
        return self._chunked_topk(query_embs, k)

    def retrieve_top_k_batch(self, all_queries: List[List[str]], k: int = 3) -> List[List[Set[int]]]:
        """
        Batch retrieval: process all queries across all examples in a single pass.
        all_queries: list of query lists, one per example (each list = [original] + perturbations)
        Returns: list of lists of doc-id sets, matching all_queries structure.
        """
        # Flatten all queries
        flat_queries = [q for qs in all_queries for q in qs]
        lengths = [len(qs) for qs in all_queries]

        logger.info(f"Encoding {len(flat_queries)} queries...")
        flat_embs = self.encode_queries(flat_queries)  # [total_Q, D]

        logger.info(f"Running chunked retrieval over {self.n_docs} docs in chunks of {self.chunk_size}...")
        flat_top_indices = self._chunked_topk(flat_embs, k)  # list of sets

        # Re-group by example
        results = []
        offset = 0
        for l in lengths:
            results.append(flat_top_indices[offset : offset + l])
            offset += l
        return results

    def retrieve_top_k_with_embs(
        self, queries: List[str], k: int = 5
    ):
        """
        Retrieve top-k docs and return structured arrays needed for DTR AIS (select_topk_of_query_info).
        Returns:
            doc_ids:   np.ndarray [Q, k] int64   — corpus doc indices
            doc_embs:  np.ndarray [Q, k, D] float32 — embeddings of retrieved docs
            scores:    np.ndarray [Q, k] float32   — inner-product scores
            query_embs: np.ndarray [Q, D] float32  — query embeddings
        """
        query_embs_t = self.encode_queries(queries)  # [Q, D]
        doc_ids_t, doc_embs_t, scores_t = self._chunked_topk_with_embs(query_embs_t, k)
        query_embs_np = query_embs_t.cpu().numpy()
        doc_ids_np = doc_ids_t.cpu().numpy().astype(np.int64)
        scores_np = scores_t.cpu().numpy().astype(np.float32)
        doc_embs_np = doc_embs_t.cpu().numpy().astype(np.float32)
        return doc_ids_np, doc_embs_np, scores_np, query_embs_np

    def _chunked_topk(self, query_embs: torch.Tensor, k: int) -> List[Set[int]]:
        Q = query_embs.shape[0]
        top_scores = torch.full((Q, k), float("-inf"), device=self.device)
        top_indices = torch.zeros((Q, k), dtype=torch.long, device=self.device)

        n_chunks = (self.n_docs + self.chunk_size - 1) // self.chunk_size
        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, self.n_docs)

            chunk = torch.from_numpy(
                np.array(self.corpus_embs[start:end], dtype=np.float32)
            ).to(self.device)  # [C, D]

            scores = torch.matmul(query_embs, chunk.t())  # [Q, C]

            combined_scores = torch.cat([top_scores, scores], dim=1)
            combined_indices = torch.cat([
                top_indices,
                torch.arange(start, end, device=self.device).unsqueeze(0).expand(Q, -1)
            ], dim=1)

            new_top_scores, sel = combined_scores.topk(k, dim=1)
            top_scores = new_top_scores
            top_indices = combined_indices.gather(1, sel)

            del chunk, scores, combined_scores, combined_indices
            torch.cuda.empty_cache()

            if (chunk_idx + 1) % 2 == 0:
                logger.info(f"  Processed {end}/{self.n_docs} docs ({100*end/self.n_docs:.1f}%)")

        top_indices_cpu = top_indices.cpu().numpy()
        return [set(int(i) for i in row) for row in top_indices_cpu]

    def _chunked_topk_with_embs(
        self, query_embs: torch.Tensor, k: int
    ):
        """Like _chunked_topk but also returns the embeddings and scores of top-k docs."""
        Q = query_embs.shape[0]
        D = query_embs.shape[1]
        top_scores = torch.full((Q, k), float("-inf"), device=self.device)
        top_indices = torch.zeros((Q, k), dtype=torch.long, device=self.device)

        n_chunks = (self.n_docs + self.chunk_size - 1) // self.chunk_size
        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, self.n_docs)

            chunk = torch.from_numpy(
                np.array(self.corpus_embs[start:end], dtype=np.float32)
            ).to(self.device)  # [C, D]

            scores = torch.matmul(query_embs, chunk.t())  # [Q, C]

            combined_scores = torch.cat([top_scores, scores], dim=1)
            combined_indices = torch.cat([
                top_indices,
                torch.arange(start, end, device=self.device).unsqueeze(0).expand(Q, -1)
            ], dim=1)

            new_top_scores, sel = combined_scores.topk(k, dim=1)
            top_scores = new_top_scores
            top_indices = combined_indices.gather(1, sel)

            del chunk, scores, combined_scores, combined_indices
            torch.cuda.empty_cache()

            if (chunk_idx + 1) % 2 == 0:
                logger.info(f"  Processed {end}/{self.n_docs} docs ({100*end/self.n_docs:.1f}%)")

        # Gather embeddings of the selected top-k docs
        top_indices_cpu = top_indices.cpu().numpy()  # [Q, k]
        flat_ids = top_indices_cpu.ravel()  # [Q*k]
        flat_embs = np.array(self.corpus_embs[flat_ids], dtype=np.float32)  # [Q*k, D]
        doc_embs_t = torch.from_numpy(flat_embs.reshape(Q, k, D))  # [Q, k, D]

        return top_indices, doc_embs_t, top_scores
