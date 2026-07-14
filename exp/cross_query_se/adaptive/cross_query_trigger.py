# Cross-query SE adaptive retrieval trigger.
# Same 3-way gating policy as se_trigger.py but uses H_cq as the uncertainty signal.
# FIX: threshold tuning now uses majority_vote_answer (not rag3) for single/enhanced branches.

import logging
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

TAU_LOW_GRID = [0.1, 0.3, 0.5, 0.7]
TAU_HIGH_GRID = [0.8, 1.0, 1.2, 1.5]


def _compute_em(pred: str, gold_answers: List[str]) -> float:
    from DTR.evaluation.metrics import exact_match_score
    return max(float(exact_match_score(pred, g)) for g in gold_answers) if gold_answers else 0.0


def apply_cq_trigger(hcq: float, tau_low: float, tau_high: float) -> str:
    """
    3-way gating using H_cq:
      H_cq < tau_low            -> 'no_retrieval'
      tau_low <= H_cq < tau_high -> 'single_retrieval'
      H_cq >= tau_high           -> 'enhanced_retrieval'
    """
    if hcq < tau_low:
        return "no_retrieval"
    elif hcq < tau_high:
        return "single_retrieval"
    else:
        return "enhanced_retrieval"


def majority_vote_answer(answers: List[str]) -> str:
    """
    Aggregate K+1 per-perturbation answers by majority vote.
    Returns the most frequent answer; breaks ties by first occurrence.
    """
    if not answers:
        return ""
    if len(answers) == 1:
        return answers[0]
    counts = Counter(answers)
    return counts.most_common(1)[0][0]


def tune_cq_thresholds(
    dev_records: List[Dict],
    tau_low_grid: List[float] = TAU_LOW_GRID,
    tau_high_grid: List[float] = TAU_HIGH_GRID,
) -> Tuple[float, float]:
    """
    Grid-search (tau_low, tau_high) maximising average EM on dev set.
    Each dev_record must have:
      - 'hcq_score'         : float
      - 'answers'           : List[str]
      - 'direct_answer'     : str
      - 'majority_answer'   : str  (majority vote from per_pert_answers)
      - 'enhanced_answer'   : Optional[str]  (not available at tuning time -> use majority)
    FIX: use majority_answer for single_retrieval and enhanced branches during tuning,
    so that the threshold actually discriminates between 'direct_answer' (no-retrieval)
    and 'majority_answer' (retrieval) rather than collapsing both to rag3_answer.
    """
    best_tau_low = tau_low_grid[0]
    best_tau_high = tau_high_grid[-1]
    best_em = -1.0

    for tau_low in tau_low_grid:
        for tau_high in tau_high_grid:
            if tau_low >= tau_high:
                continue
            total_em = 0.0
            for rec in dev_records:
                hcq = rec["hcq_score"]
                answers = rec["answers"]
                decision = apply_cq_trigger(hcq, tau_low, tau_high)
                if decision == "no_retrieval":
                    pred = rec["direct_answer"]
                elif decision == "single_retrieval":
                    # Use majority_answer if available, else rag3
                    pred = rec.get("majority_answer") or rec.get("rag3_answer", "")
                else:
                    # enhanced_answer not available at tuning time; use majority as proxy
                    pred = rec.get("majority_answer") or rec.get("rag3_answer", "")
                total_em += _compute_em(pred, answers)
            avg_em = total_em / len(dev_records) if dev_records else 0.0
            if avg_em > best_em:
                best_em = avg_em
                best_tau_low = tau_low
                best_tau_high = tau_high

    logger.info(
        f"Best thresholds: tau_low={best_tau_low}, tau_high={best_tau_high}, dev_EM={best_em:.4f}"
    )
    return best_tau_low, best_tau_high


def select_best_answer_by_relevance(
    answers: List[str],
    retrieval_scores: List[float],
) -> str:
    """
    Select the answer from the perturbation whose retrieved documents have the
    highest relevance score to the original query.
    NOTE: This is kept for backward compatibility but majority_vote_answer is preferred.
    """
    if not answers:
        return ""
    if len(answers) == 1:
        return answers[0]
    best_idx = int(np.argmax(retrieval_scores))
    return answers[best_idx]
