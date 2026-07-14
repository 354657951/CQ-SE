# Mechanism hypothesis analysis for task 13.
# Step 1: Stratify queries by V_ret (low/medium/high tercile) and compute per-stratum AUROC
#         for cross-query SE, within-query SE (SUGAR), INTRYGUE, and Token-NLL (DTR).
# Step 2: Spearman correlation between per-query V_ret and AUROC advantage (cross-query - within-query).
# Step 3: Generate visualizations (bar chart, scatter+regression, sanity-check table).
# Step 4: Write results to EXPERIMENT_RESULTS/task_13/.

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ["nq", "webqa", "triviaqa", "hotpotqa", "squad"]


def _jsonl_load(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def compute_vret(per_query_doc_ids_list: List[List[List[int]]]) -> float:
    """Compute V_ret for one query = average pairwise Jaccard distance over K perturbation doc sets."""
    # per_query_doc_ids_list: list of doc_id lists, first is original query
    sets = [set(ids) for ids in per_query_doc_ids_list]
    if len(sets) <= 1:
        return 0.0
    dists = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            union = a | b
            if not union:
                dists.append(0.0)
            else:
                jaccard = len(a & b) / len(union)
                dists.append(1.0 - jaccard)
    return float(np.mean(dists)) if dists else 0.0


def load_per_query_data(
    datasets: List[str],
    cq_dir: str,
    sugar_dir: str,
    intrygue_dir: str,
    token_nll_dir: Optional[str],
    dev_size: int = 500,
) -> Dict[str, List[Dict]]:
    """Load per-query scores and labels for all methods, averaged across seeds 0/1/2."""
    result = {}
    for ds in datasets:
        # ── Ground truth (questions, answers, correctness) ───────────────────
        base_path = os.path.join(cq_dir, f"{ds}_vllm_base.jsonl")
        base_records = _jsonl_load(base_path)
        test_base = base_records[dev_size:]
        q_to_answers = {r["question"]: r["answers"] for r in test_base}
        q_to_rag3_answer = {r["question"]: r["rag3_answer"] for r in test_base}
        all_questions = [r["question"] for r in test_base]

        # ── Load per-seed data ────────────────────────────────────────────────
        seeds = [0, 1, 2]
        # per-seed V_ret, H_cq, SE scores
        vret_by_seed: Dict[int, Dict[str, float]] = {}
        hcq_by_seed: Dict[int, Dict[str, float]] = {}
        se_by_seed: Dict[int, Dict[str, float]] = {}

        for seed in seeds:
            # V_ret from pert_retrieval
            pert_ret_path = os.path.join(cq_dir, f"{ds}_pert_retrieval_seed{seed}.jsonl")
            if os.path.exists(pert_ret_path):
                vret_by_seed[seed] = {}
                for r in _jsonl_load(pert_ret_path):
                    vret = compute_vret(r["per_query_doc_ids"])
                    vret_by_seed[seed][r["question"]] = vret
            else:
                logger.warning(f"Missing {pert_ret_path}")

            # H_cq from hcq records
            hcq_path = os.path.join(cq_dir, f"{ds}_hcq_seed{seed}.jsonl")
            if os.path.exists(hcq_path):
                hcq_by_seed[seed] = {}
                for r in _jsonl_load(hcq_path):
                    hcq_by_seed[seed][r["question"]] = r["hcq_score"]
            else:
                logger.warning(f"Missing {hcq_path}")

            # Within-query SE from SUGAR
            se_path = os.path.join(sugar_dir, f"{ds}_se_seed{seed}.jsonl")
            if os.path.exists(se_path):
                se_by_seed[seed] = {}
                for r in _jsonl_load(se_path):
                    se_by_seed[seed][r["question"]] = r["se_score"]
            else:
                logger.warning(f"Missing {se_path}")

        # ── Load INTRYGUE (seed-independent) ─────────────────────────────────
        intrygue_path = os.path.join(intrygue_dir, f"{ds}_intrygue_scores.jsonl")
        intrygue_scores: Dict[str, float] = {}
        if os.path.exists(intrygue_path):
            for r in _jsonl_load(intrygue_path):
                intrygue_scores[r["question"]] = r.get("intrygue_mean", 0.0)
        else:
            logger.warning(f"Missing {intrygue_path}")

        # ── Load Token-NLL ────────────────────────────────────────────────────
        token_nll_scores: Dict[str, float] = {}
        if token_nll_dir:
            tnll_path = os.path.join(token_nll_dir, f"{ds}_token_nll.jsonl")
            if os.path.exists(tnll_path):
                for r in _jsonl_load(tnll_path):
                    token_nll_scores[r["question"]] = r["token_nll"]
            else:
                logger.warning(f"Missing token-NLL file: {tnll_path}")

        # ── Aggregate per query ───────────────────────────────────────────────
        from DTR.evaluation.metrics import exact_match_score, f1_score

        def _em(pred: str, golds: List[str]) -> float:
            return max((float(exact_match_score(pred, g)) for g in golds), default=0.0)

        def _f1(pred: str, golds: List[str]) -> float:
            return max((f1_score(pred, g)[0] for g in golds), default=0.0)

        query_data = []
        for q in all_questions:
            answers = q_to_answers.get(q, [])
            rag3_ans = q_to_rag3_answer.get(q, "")
            correct = int(_em(rag3_ans, answers) > 0)

            # Average V_ret across seeds
            vret_vals = [vret_by_seed[s][q] for s in seeds if s in vret_by_seed and q in vret_by_seed[s]]
            avg_vret = float(np.mean(vret_vals)) if vret_vals else 0.0

            # Average H_cq across seeds
            hcq_vals = [hcq_by_seed[s][q] for s in seeds if s in hcq_by_seed and q in hcq_by_seed[s]]
            avg_hcq = float(np.mean(hcq_vals)) if hcq_vals else 0.0

            # Average SE across seeds
            se_vals = [se_by_seed[s][q] for s in seeds if s in se_by_seed and q in se_by_seed[s]]
            avg_se = float(np.mean(se_vals)) if se_vals else 0.0

            intrygue_val = intrygue_scores.get(q, 0.0)
            token_nll_val = token_nll_scores.get(q, None)

            query_data.append({
                "question": q,
                "correct": correct,
                "vret": avg_vret,
                "hcq": avg_hcq,
                "se": avg_se,
                "intrygue": intrygue_val,
                "token_nll": token_nll_val,
            })

        result[ds] = query_data
        logger.info(f"[{ds}] Loaded {len(query_data)} queries, "
                    f"avg V_ret={np.mean([q['vret'] for q in query_data]):.4f}, "
                    f"correct={np.mean([q['correct'] for q in query_data]):.4f}")

    return result


def compute_auroc(scores: List[float], labels: List[int]) -> float:
    """Compute AUROC, returns nan if degenerate."""
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2 or len(scores) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def compute_em_f1(query_data: List[Dict]) -> Tuple[float, float]:
    corrects = [q["correct"] for q in query_data]
    return float(np.mean(corrects)), float(np.mean(corrects))  # EM proxy


def stratify_analysis(query_data_by_ds: Dict[str, List[Dict]]) -> Dict:
    """Step 1: stratify by V_ret tercile, compute per-stratum AUROC for all methods."""
    results = {}
    for ds, query_data in query_data_by_ds.items():
        vrets = np.array([q["vret"] for q in query_data])
        t33 = float(np.percentile(vrets, 33.33))
        t66 = float(np.percentile(vrets, 66.67))

        strata = {"low": [], "medium": [], "high": []}
        for q in query_data:
            if q["vret"] <= t33:
                strata["low"].append(q)
            elif q["vret"] <= t66:
                strata["medium"].append(q)
            else:
                strata["high"].append(q)

        ds_results = {"thresholds": {"t33": t33, "t66": t66}, "strata": {}}
        for stratum_name, stratum_qs in strata.items():
            if not stratum_qs:
                ds_results["strata"][stratum_name] = {}
                continue
            labels = [q["correct"] for q in stratum_qs]
            # AUROC: higher score = more confident = more likely correct
            auroc_cq = compute_auroc([-q["hcq"] for q in stratum_qs], labels)
            auroc_se = compute_auroc([-q["se"] for q in stratum_qs], labels)
            auroc_intrygue = compute_auroc([-q["intrygue"] for q in stratum_qs], labels)
            tnll_scores = [q["token_nll"] for q in stratum_qs if q["token_nll"] is not None]
            tnll_labels = [q["correct"] for q in stratum_qs if q["token_nll"] is not None]
            # Higher token_nll = less confident = more uncertain; negate for AUROC
            auroc_tnll = compute_auroc([-s for s in tnll_scores], tnll_labels) if tnll_scores else float("nan")

            avg_vret = float(np.mean([q["vret"] for q in stratum_qs]))
            avg_correct = float(np.mean(labels))

            ds_results["strata"][stratum_name] = {
                "n": len(stratum_qs),
                "avg_vret": avg_vret,
                "avg_correct": avg_correct,
                "auroc_cross_query_se": auroc_cq,
                "auroc_within_query_se": auroc_se,
                "auroc_intrygue": auroc_intrygue,
                "auroc_token_nll": auroc_tnll,
            }
            logger.info(
                f"[{ds}] {stratum_name} (n={len(stratum_qs)}, vret={avg_vret:.3f}): "
                f"CQ-SE={auroc_cq:.4f} WQ-SE={auroc_se:.4f} INTRYGUE={auroc_intrygue:.4f} "
                f"TokenNLL={auroc_tnll:.4f}"
            )
        results[ds] = ds_results
    return results


def spearman_analysis(query_data_by_ds: Dict[str, List[Dict]]) -> Dict:
    """Step 2: Spearman correlation between V_ret and per-query AUROC advantage."""
    # Since AUROC is aggregate, use per-query correctness advantage proxy:
    # advantage[q] = I(CQ predicts correctly) - I(SE predicts correctly)
    # where CQ predicts correctly = lower H_cq than the stratum median (sign-consistent with correctness)
    # More precisely: use signed confidence scores: -H_cq (higher = more confident, cross-query)
    # vs -SE (higher = more confident, within-query), and measure alignment with ground truth.
    # Actually: measure correlation between V_ret and |advantage| in confidence alignment.
    # Simpler approach: per-query "contribution to AUROC" = correct * (-H_cq) - correct * (-SE)
    # = correct * (SE - H_cq); positive when cross-query is more confident on correct answers.
    # Best well-defined metric: correlation between V_ret and (conf_cq - conf_se) * (2*correct - 1)
    # i.e., how much better CQ is at assigning confidence that aligns with correctness.
    results = {}
    for ds, query_data in query_data_by_ds.items():
        vrets = np.array([q["vret"] for q in query_data])
        # Normalized confidence scores (negated entropy for "higher = more confident")
        conf_cq = np.array([-q["hcq"] for q in query_data])
        conf_se = np.array([-q["se"] for q in query_data])
        labels = np.array([q["correct"] for q in query_data])

        # Per-query advantage: difference in "correct-aligned confidence"
        advantage = (conf_cq - conf_se) * (2 * labels - 1)

        corr, pval = stats.spearmanr(vrets, advantage)
        results[ds] = {
            "spearman_r": float(corr),
            "p_value": float(pval),
            "n": len(query_data),
        }
        logger.info(f"[{ds}] Spearman(V_ret, advantage): r={corr:.4f}, p={pval:.4f} (n={len(query_data)})")

    return results


def load_random_pert_results(random_pert_results_dir: str, datasets: List[str]) -> Dict:
    """Load AUROC from random perturbation sanity check results."""
    results_path = os.path.join(random_pert_results_dir, "random_pert_sanity_results.json")
    if not os.path.exists(results_path):
        logger.warning(f"Random perturbation results not found: {results_path}")
        return {}
    with open(results_path) as f:
        return json.load(f)


def load_existing_method_auroc(datasets: List[str]) -> Dict:
    """Load overall AUROC from the main cross-query SE and sugar baseline results."""
    cq_results_path = "cross_query_se/results/cross_query_se_opt/cross_query_se_results.json"
    sugar_results_path = "cross_query_se/results/sugar_baseline/sugar_baseline_results.json"

    cq_all = {}
    sugar_all = {}
    if os.path.exists(cq_results_path):
        with open(cq_results_path) as f:
            d = json.load(f)
        for ds in datasets:
            if ds in d:
                cq_all[ds] = d[ds].get("auroc_mean", float("nan"))
    if os.path.exists(sugar_results_path):
        with open(sugar_results_path) as f:
            d = json.load(f)
        for ds in datasets:
            if ds in d:
                sugar_all[ds] = d[ds].get("auroc_mean", float("nan"))
    return {"cross_query_se": cq_all, "within_query_se": sugar_all}


def make_visualizations(
    stratify_results: Dict,
    spearman_results: Dict,
    query_data_by_ds: Dict[str, List[Dict]],
    random_pert_results: Dict,
    existing_auroc: Dict,
    figures_dir: str,
    datasets: List[str],
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(figures_dir, exist_ok=True)

    # ── Fig 4a: Grouped bar chart AUROC × stratum × method ───────────────────
    n_datasets = len(datasets)
    stratum_names = ["low", "medium", "high"]
    method_keys = [
        ("auroc_cross_query_se", "Cross-Query SE", "steelblue"),
        ("auroc_within_query_se", "Within-Query SE", "coral"),
        ("auroc_intrygue", "INTRYGUE", "mediumseagreen"),
        ("auroc_token_nll", "Token-NLL (DTR)", "mediumpurple"),
    ]
    n_strata = len(stratum_names)
    n_methods = len(method_keys)
    bar_width = 0.18
    x = np.arange(n_strata)

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        ds_strata = stratify_results.get(ds, {}).get("strata", {})
        for m_idx, (key, label, color) in enumerate(method_keys):
            vals = []
            for sn in stratum_names:
                v = ds_strata.get(sn, {}).get(key, float("nan"))
                vals.append(v if not np.isnan(v) else 0.0)
            offset = (m_idx - (n_methods - 1) / 2) * bar_width
            bars = ax.bar(x + offset, vals, bar_width, label=label, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xlabel("V_ret Stratum")
        ax.set_ylabel("AUROC")
        ax.set_title(f"AUROC by V_ret Stratum — {ds.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(["Low V_ret", "Medium V_ret", "High V_ret"])
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"fig4a_auroc_stratum_{ds}.png"), dpi=150)
        plt.close(fig)
        logger.info(f"Saved fig4a for {ds}")

    # ── Fig 4a (aggregate): all datasets side-by-side ───────────────────────
    fig, axes = plt.subplots(1, n_datasets, figsize=(4 * n_datasets, 5), sharey=True)
    if n_datasets == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        ds_strata = stratify_results.get(ds, {}).get("strata", {})
        for m_idx, (key, label, color) in enumerate(method_keys):
            vals = [ds_strata.get(sn, {}).get(key, float("nan")) for sn in stratum_names]
            vals = [v if not np.isnan(v) else 0.0 for v in vals]
            offset = (m_idx - (n_methods - 1) / 2) * bar_width
            ax.bar(x + offset, vals, bar_width, label=label, color=color, alpha=0.85)
        ax.set_xlabel(ds.upper())
        ax.set_title(ds.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(["Low", "Med", "High"], fontsize=9)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    handles = [mpatches.Patch(color=c, label=l) for (_, l, c) in method_keys]
    fig.legend(handles=handles, loc="upper center", ncol=n_methods, fontsize=9, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.04, 0.5, "AUROC", va="center", rotation="vertical")
    fig.suptitle("AUROC by V_ret Stratum (All Datasets)", y=1.06)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "fig4a_auroc_stratum_all.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fig4a (all datasets)")

    # ── Fig 4b: Scatter V_ret vs AUROC advantage, with regression line ────────
    for ds in datasets:
        query_data = query_data_by_ds[ds]
        vrets = np.array([q["vret"] for q in query_data])
        labels = np.array([q["correct"] for q in query_data])
        conf_cq = np.array([-q["hcq"] for q in query_data])
        conf_se = np.array([-q["se"] for q in query_data])
        advantage = (conf_cq - conf_se) * (2 * labels - 1)

        fig, ax = plt.subplots(figsize=(6, 5))
        # Use hexbin for large N
        hb = ax.hexbin(vrets, advantage, gridsize=40, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count")

        # Regression line
        slope, intercept, r_val, p_val, _ = stats.linregress(vrets, advantage)
        x_line = np.linspace(vrets.min(), vrets.max(), 100)
        ax.plot(x_line, intercept + slope * x_line, "r-", linewidth=2,
                label=f"r={r_val:.3f}, p={p_val:.3f}")
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        sr = spearman_results.get(ds, {})
        ax.set_xlabel("V_ret (Retrieval Variance)")
        ax.set_ylabel("CQ-SE Advantage (correctness-aligned)")
        ax.set_title(f"V_ret vs. CQ-SE Advantage — {ds.upper()}\n"
                     f"Spearman r={sr.get('spearman_r', 0):.3f}, p={sr.get('p_value', 1):.3f}")
        ax.legend(fontsize=9)
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"fig4b_scatter_{ds}.png"), dpi=150)
        plt.close(fig)
        logger.info(f"Saved fig4b for {ds}")

    # ── Fig 4c: Summary table as image ────────────────────────────────────────
    # Table: AUROC comparison cross-query SE, random-pert entropy, within-query SE
    if random_pert_results:
        col_labels = ["Dataset", "CQ-SE (meaningful)", "Random-Pert Entropy", "Within-Query SE"]
        table_data = []
        for ds in datasets:
            cq_auroc = existing_auroc.get("cross_query_se", {}).get(ds, float("nan"))
            rand_auroc = random_pert_results.get(ds, {}).get("auroc_mean", float("nan"))
            wq_auroc = existing_auroc.get("within_query_se", {}).get(ds, float("nan"))
            table_data.append([
                ds.upper(),
                f"{cq_auroc:.4f}" if not np.isnan(cq_auroc) else "N/A",
                f"{rand_auroc:.4f}" if not np.isnan(rand_auroc) else "N/A",
                f"{wq_auroc:.4f}" if not np.isnan(wq_auroc) else "N/A",
            ])

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        tbl = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)
        ax.set_title("Fig 4c: AUROC — Meaningful vs. Random Perturbations vs. Within-Query SE")
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "fig4c_sanity_table.png"), dpi=150)
        plt.close(fig)
        logger.info("Saved fig4c")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--cq_dir", default="cross_query_se/outputs/cross_query_se")
    parser.add_argument("--sugar_dir", default="cross_query_se/outputs/sugar_baseline")
    parser.add_argument("--intrygue_dir", default="cross_query_se/outputs/intrygue_baseline")
    parser.add_argument("--token_nll_dir", default="cross_query_se/outputs/token_nll")
    parser.add_argument("--random_pert_dir", default="cross_query_se/results/random_pert_sanity")
    parser.add_argument("--results_dir", default="EXPERIMENT_RESULTS/task_13")
    parser.add_argument("--dev_size", type=int, default=500)
    args = parser.parse_args()

    figures_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ── Load per-query data ────────────────────────────────────────────────────
    logger.info("Loading per-query data...")
    query_data_by_ds = load_per_query_data(
        args.datasets, args.cq_dir, args.sugar_dir, args.intrygue_dir,
        args.token_nll_dir if os.path.isdir(args.token_nll_dir) else None,
        args.dev_size,
    )

    # ── Step 1: V_ret stratification ─────────────────────────────────────────
    logger.info("=== Step 1: V_ret Stratification Analysis ===")
    stratify_results = stratify_analysis(query_data_by_ds)

    # ── Step 2: Spearman correlation ──────────────────────────────────────────
    logger.info("=== Step 2: Spearman Correlation ===")
    spearman_results = spearman_analysis(query_data_by_ds)

    # ── Load random perturbation results & existing AUROC ─────────────────────
    random_pert_results = load_random_pert_results(args.random_pert_dir, args.datasets)
    existing_auroc = load_existing_method_auroc(args.datasets)

    # ── Step 3: Visualizations ────────────────────────────────────────────────
    logger.info("=== Step 3: Visualizations ===")
    make_visualizations(
        stratify_results, spearman_results, query_data_by_ds,
        random_pert_results, existing_auroc, figures_dir, args.datasets,
    )

    # ── Compile full results JSON ─────────────────────────────────────────────
    results = {
        "task": "Mechanism Hypothesis Validation",
        "task_index": 13,
        "step1_vret_stratification": stratify_results,
        "step2_spearman_correlation": spearman_results,
        "step3_random_perturbation_sanity": {
            "random_pert_results": random_pert_results,
            "existing_overall_auroc": existing_auroc,
            "description": (
                "Random perturbation entropy compared to meaningful cross-query SE and within-query SE. "
                "Expected: random-pert AUROC ~ 0.5 (no discriminative power)."
            ),
        },
        "mechanism_verdict": _compute_verdict(stratify_results, spearman_results),
    }

    results_path = os.path.join(args.results_dir, "RESULTS.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n=== MECHANISM HYPOTHESIS SUMMARY ===")
    for ds in args.datasets:
        logger.info(f"\n{ds.upper()}:")
        strata = stratify_results.get(ds, {}).get("strata", {})
        for sn in ["low", "medium", "high"]:
            s = strata.get(sn, {})
            n = s.get("n", 0)
            cq = s.get("auroc_cross_query_se", float("nan"))
            se = s.get("auroc_within_query_se", float("nan"))
            it = s.get("auroc_intrygue", float("nan"))
            tn = s.get("auroc_token_nll", float("nan"))
            logger.info(f"  {sn:6s} (n={n:4d}): CQ-SE={cq:.4f} WQ-SE={se:.4f} INTRYGUE={it:.4f} TokenNLL={tn:.4f}")
        sp = spearman_results.get(ds, {})
        logger.info(f"  Spearman(V_ret, advantage): r={sp.get('spearman_r', 0):.4f}, p={sp.get('p_value', 1):.4f}")


def _compute_verdict(stratify_results: Dict, spearman_results: Dict) -> Dict:
    """Assess whether the mechanism hypothesis is supported."""
    verdict = {}
    for ds, strata_info in stratify_results.items():
        strata = strata_info.get("strata", {})
        low_cq = strata.get("low", {}).get("auroc_cross_query_se", float("nan"))
        high_cq = strata.get("high", {}).get("auroc_cross_query_se", float("nan"))
        low_se = strata.get("low", {}).get("auroc_within_query_se", float("nan"))
        high_se = strata.get("high", {}).get("auroc_within_query_se", float("nan"))

        # CQ-SE should improve more from low to high V_ret than WQ-SE
        cq_improvement = (high_cq - low_cq) if not (np.isnan(high_cq) or np.isnan(low_cq)) else float("nan")
        se_improvement = (high_se - low_se) if not (np.isnan(high_se) or np.isnan(low_se)) else float("nan")
        hypothesis_supported = (
            (not np.isnan(cq_improvement) and not np.isnan(se_improvement) and cq_improvement > se_improvement)
            and spearman_results.get(ds, {}).get("spearman_r", 0) > 0
        )

        verdict[ds] = {
            "cq_auroc_improvement_low_to_high": cq_improvement,
            "se_auroc_improvement_low_to_high": se_improvement,
            "hypothesis_supported": hypothesis_supported,
            "spearman_r": spearman_results.get(ds, {}).get("spearman_r", float("nan")),
        }
    return verdict


if __name__ == "__main__":
    main()
