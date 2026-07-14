# SUGAR-style SE-based adaptive retrieval trigger.
# Tunes tau_low / tau_high on a held-out dev set, then applies the 3-way gating policy.

import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

TAU_LOW_GRID = [0.1, 0.3, 0.5, 0.7]
TAU_HIGH_GRID = [0.8, 1.0, 1.2, 1.5]


def _compute_em(pred: str, gold_answers: List[str]) -> float:
    """EM over a list of valid gold answers (max)."""
    from DTR.evaluation.metrics import exact_match_score
    return max(float(exact_match_score(pred, g)) for g in gold_answers) if gold_answers else 0.0


def apply_se_trigger(se: float, tau_low: float, tau_high: float) -> str:
    """
    3-way gating:
      SE < tau_low            -> 'no_retrieval'
      tau_low <= SE < tau_high -> 'single_retrieval'
      SE >= tau_high           -> 'enhanced_retrieval'
    """
    if se < tau_low:
        return "no_retrieval"
    elif se < tau_high:
        return "single_retrieval"
    else:
        return "enhanced_retrieval"


def tune_se_thresholds(
    dev_records: List[Dict],
    tau_low_grid: List[float] = TAU_LOW_GRID,
    tau_high_grid: List[float] = TAU_HIGH_GRID,
) -> Tuple[float, float]:
    """
    Grid-search (tau_low, tau_high) maximising average EM on the dev set.
    Each dev_record must have:
      - 'se_score'          : float  (SE computed from M=5 samples)
      - 'answers'           : List[str]
      - 'direct_answer'     : str  (greedy, no retrieval)
      - 'rag3_answer'       : str  (greedy, top-3 RAG)
      - 'enhanced_answer'   : str  (DTR dual-path AIS answer; may be None if not pre-computed)
    Note: enhanced_answer is often None during dev threshold search because AIS requires
    a second retrieval+generation pass. We fall back to rag3_answer for enhanced_retrieval
    during threshold tuning, which is acceptable since we only need the relative ordering.
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
                se = rec["se_score"]
                answers = rec["answers"]
                decision = apply_se_trigger(se, tau_low, tau_high)
                if decision == "no_retrieval":
                    pred = rec["direct_answer"]
                elif decision == "single_retrieval":
                    pred = rec["rag3_answer"]
                else:
                    # Fall back to rag3 during tuning if enhanced not available
                    pred = rec.get("enhanced_answer") or rec["rag3_answer"]
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
