# Retrieval variance V_ret computation and dataset-level analysis.
# V_ret(x) = mean pairwise Jaccard distance across retrieved doc sets for x and its perturbations.

import os
import logging
from itertools import combinations
from typing import List, Set, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def jaccard_distance(set_a: Set, set_b: Set) -> float:
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - intersection / union if union > 0 else 0.0


def compute_v_ret(doc_id_sets: List[Set]) -> float:
    if len(doc_id_sets) < 2:
        return 0.0
    pairs = list(combinations(doc_id_sets, 2))
    dists = [jaccard_distance(a, b) for a, b in pairs]
    return float(np.mean(dists))


def compute_dataset_stats(v_ret_values: List[float]) -> Dict[str, float]:
    arr = np.array(v_ret_values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "q90": float(np.percentile(arr, 90)),
        "n": len(arr),
        "n_nonzero": int(np.sum(arr > 0)),
        "frac_nonzero": float(np.mean(arr > 0)),
        "frac_above_005": float(np.mean(arr >= 0.05)),
        "frac_above_01": float(np.mean(arr >= 0.1)),
    }


def plot_v_ret_histograms(
    dataset_v_rets: Dict[str, List[float]],
    output_path: str,
    feasibility_threshold: float = 0.05,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        datasets = list(dataset_v_rets.keys())
        n = len(datasets)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]

        for ax, ds in zip(axes, datasets):
            vals = dataset_v_rets[ds]
            sns.histplot(vals, bins=30, ax=ax, color="steelblue", edgecolor="white", alpha=0.8)
            median_v = float(np.median(vals))
            ax.axvline(median_v, color="red", linestyle="--", linewidth=1.5, label=f"Median={median_v:.3f}")
            ax.axvline(feasibility_threshold, color="orange", linestyle=":", linewidth=1.5, label=f"Gate={feasibility_threshold}")
            ax.set_title(ds.upper(), fontsize=12)
            ax.set_xlabel("V_ret (Jaccard distance)", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.legend(fontsize=8)

        plt.suptitle("Retrieval Variance V_ret Distribution per Dataset", fontsize=13)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved V_ret histograms to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to plot histograms: {e}")


def apply_feasibility_gate(
    dataset_stats: Dict[str, Dict[str, float]],
    threshold: float = 0.05,
    min_datasets: int = 2,
) -> Dict[str, Any]:
    passing = [ds for ds, stats in dataset_stats.items() if stats["median"] >= threshold]
    gate_pass = len(passing) >= min_datasets
    return {
        "gate_pass": gate_pass,
        "threshold": threshold,
        "min_datasets": min_datasets,
        "passing_datasets": passing,
        "n_passing": len(passing),
        "verdict": "GO" if gate_pass else "NO-GO",
    }
