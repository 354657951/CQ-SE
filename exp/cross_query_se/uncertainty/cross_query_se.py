# Cross-query semantic entropy (H_cq) estimator.
# Clusters K+1 answers (original + K perturbation-driven greedy answers) using bidirectional
# DeBERTa entailment, then computes H_cq = -sum_c p(c)*log(p(c)).
# Inherits WithinQuerySE for NLI batching and greedy clustering.

import math
import logging
from collections import Counter
from typing import List, Tuple

from cross_query_se.uncertainty.within_query_se import WithinQuerySE

logger = logging.getLogger(__name__)


class CrossQuerySE(WithinQuerySE):
    def compute_cross_query_se(
        self, answers: List[str]
    ) -> Tuple[float, List[int]]:
        """
        Compute H_cq for a single set of K+1 answers.
        Returns (entropy, cluster_ids).
        """
        N = len(answers)
        if N == 0:
            return 0.0, []
        if N == 1:
            return 0.0, [0]
        matrix = self._build_entail_matrix(answers)
        cluster_ids = self._greedy_cluster(answers, matrix)
        counts = Counter(cluster_ids)
        H = 0.0
        for cnt in counts.values():
            p = cnt / N
            H -= p * math.log(p + 1e-10)
        return H, cluster_ids

    def compute_cross_query_se_batch(
        self, all_answers: List[List[str]]
    ) -> List[Tuple[float, List[int]]]:
        """
        Batched H_cq computation for a list of K+1 answer sets.
        Uses a single large DeBERTa pass over all pairs (same strategy as compute_se_batch).
        Returns list of (entropy, cluster_ids) tuples.
        """
        all_pairs: List[Tuple[str, str]] = []
        pair_offsets: List[int] = []
        pair_counts: List[int] = []

        for answers in all_answers:
            N = len(answers)
            start = len(all_pairs)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        all_pairs.append((answers[i], answers[j]))
            pair_offsets.append(start)
            pair_counts.append(len(all_pairs) - start)

        if all_pairs:
            all_results = self._run_nli_batch(all_pairs)
        else:
            all_results = []

        outputs = []
        for q_idx, answers in enumerate(all_answers):
            N = len(answers)
            if N == 0:
                outputs.append((0.0, []))
                continue
            if N == 1:
                outputs.append((0.0, [0]))
                continue

            offset = pair_offsets[q_idx]
            count = pair_counts[q_idx]
            raw = all_results[offset : offset + count]

            matrix = [[False] * N for _ in range(N)]
            k = 0
            for i in range(N):
                for j in range(N):
                    if i != j:
                        matrix[i][j] = raw[k]
                        k += 1
            for i in range(N):
                matrix[i][i] = True

            cluster_ids = self._greedy_cluster(answers, matrix)
            counts = Counter(cluster_ids)
            H = 0.0
            for cnt in counts.values():
                p = cnt / N
                H -= p * math.log(p + 1e-10)
            outputs.append((H, cluster_ids))

        return outputs
