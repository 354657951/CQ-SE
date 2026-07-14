# Within-query semantic entropy (SUGAR-style) for adaptive retrieval triggering.
# Implements Kuhn et al. (2023) greedy clustering with bidirectional DeBERTa NLI entailment.
# SE = -sum_c p(c) * log(p(c)) where p(c) = |cluster c| / M.

import math
import logging
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

ENTAILMENT_IDX = 2


class WithinQuerySE:
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v2-xlarge-mnli",
        device: str = "cuda:0",
        batch_size: int = 128,
        cache_dir: str = None,
    ):
        self.device = device
        self.batch_size = batch_size
        logger.info(f"Loading NLI model {model_name} on {device}")
        self.nli_tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=cache_dir
        ).to(device).eval()

    def _run_nli_batch(self, pairs: List[Tuple[str, str]]) -> List[bool]:
        """Run NLI on a list of (premise, hypothesis) pairs. Returns list of bool (entailment)."""
        results = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i : i + self.batch_size]
            premises = [p for p, _ in chunk]
            hypotheses = [h for _, h in chunk]
            enc = self.nli_tok(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.nli_model(**enc).logits  # [B, 3]
            preds = logits.argmax(dim=-1).cpu().tolist()
            results.extend([p == ENTAILMENT_IDX for p in preds])
        return results

    def _build_entail_matrix(self, answers: List[str]) -> List[List[bool]]:
        """Build M×M bidirectional entailment matrix for a single query's answers."""
        M = len(answers)
        pairs = []
        pair_coords = []
        for i in range(M):
            for j in range(M):
                if i != j:
                    pairs.append((answers[i], answers[j]))
                    pair_coords.append((i, j))
        raw = self._run_nli_batch(pairs)
        matrix = [[False] * M for _ in range(M)]
        for (i, j), val in zip(pair_coords, raw):
            matrix[i][j] = val
        for i in range(M):
            matrix[i][i] = True
        return matrix

    def _greedy_cluster(self, answers: List[str], matrix: List[List[bool]]) -> List[int]:
        """
        Greedy clustering (Kuhn et al. 2023):
        For each answer, assign to the first cluster whose representative bidirectionally
        entails the candidate. Bidirectional: matrix[rep][cand] AND matrix[cand][rep].
        Returns cluster assignment list of length M.
        """
        M = len(answers)
        cluster_ids = [-1] * M
        cluster_reps = []
        next_cluster = 0
        for i in range(M):
            assigned = False
            for c, rep in enumerate(cluster_reps):
                if matrix[rep][i] and matrix[i][rep]:
                    cluster_ids[i] = c
                    assigned = True
                    break
            if not assigned:
                cluster_ids[i] = next_cluster
                cluster_reps.append(i)
                next_cluster += 1
        return cluster_ids

    def compute_se(self, answers: List[str]) -> float:
        """Compute discrete SE for a single query given M sampled answers."""
        M = len(answers)
        if M == 0:
            return 0.0
        if M == 1:
            return 0.0
        matrix = self._build_entail_matrix(answers)
        cluster_ids = self._greedy_cluster(answers, matrix)
        counts: dict = {}
        for c in cluster_ids:
            counts[c] = counts.get(c, 0) + 1
        se = 0.0
        for cnt in counts.values():
            p = cnt / M
            se -= p * math.log(p + 1e-10)
        return se

    def compute_se_batch(self, all_answers: List[List[str]]) -> List[float]:
        """
        Compute SE for a batch of queries efficiently.
        all_answers: list of M-length answer lists, one per query.
        Batches all NLI pairs across the dataset into a single large DeBERTa pass.
        """
        N = len(all_answers)
        # Build all pairs globally
        all_pairs: List[Tuple[str, str]] = []
        pair_offsets: List[int] = []  # starting index in all_pairs for each query
        pair_counts: List[int] = []   # number of pairs for each query

        for answers in all_answers:
            M = len(answers)
            start = len(all_pairs)
            for i in range(M):
                for j in range(M):
                    if i != j:
                        all_pairs.append((answers[i], answers[j]))
            pair_offsets.append(start)
            pair_counts.append(len(all_pairs) - start)

        # Single batched NLI pass
        if all_pairs:
            all_results = self._run_nli_batch(all_pairs)
        else:
            all_results = []

        # Reconstruct SE per query
        se_scores = []
        for q_idx, answers in enumerate(all_answers):
            M = len(answers)
            if M <= 1:
                se_scores.append(0.0)
                continue
            offset = pair_offsets[q_idx]
            count = pair_counts[q_idx]
            raw = all_results[offset : offset + count]
            # Reconstruct M×M matrix
            matrix = [[False] * M for _ in range(M)]
            k = 0
            for i in range(M):
                for j in range(M):
                    if i != j:
                        matrix[i][j] = raw[k]
                        k += 1
            for i in range(M):
                matrix[i][i] = True
            cluster_ids = self._greedy_cluster(answers, matrix)
            counts: dict = {}
            for c in cluster_ids:
                counts[c] = counts.get(c, 0) + 1
            se = 0.0
            for cnt in counts.values():
                p = cnt / M
                se -= p * math.log(p + 1e-10)
            se_scores.append(se)

        return se_scores
