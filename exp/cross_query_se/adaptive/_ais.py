# DTR AIS (Answer-Informed Selection): select top-k docs from dual-path retrieval results.
# Extracted from DTR/agent/reranker.py (select_topk_of_query_info), removing modelscope dep.

from typing import List
from collections import defaultdict

import numpy as np


def select_topk_of_query_info(
    doc_ids_query,
    doc_ids_info,
    doc_embeddings_query,
    doc_embeddings_info,
    batch_D_query,
    batch_D_info,
    query_embeddings,
    info_embeddings,
    topk_new,
    consider_adaptive: bool = False,
):
    """
    Select top-k docs from the union of query-retrieved and info-retrieved docs.
    Score = s1 + s2 (inner products with query and info embeddings).
    Missing scores are computed dynamically via inner product with stored doc embeddings.
    """
    batch_size = len(doc_ids_query)
    selected_doc_ids = []
    selected_scores = []

    for i in range(batch_size):
        doc_to_data = defaultdict(dict)

        for doc_id, emb, s1 in zip(doc_ids_query[i], doc_embeddings_query[i], batch_D_query[i]):
            doc_to_data[doc_id]["s1"] = s1
            doc_to_data[doc_id]["emb"] = emb

        for doc_id, emb, s2 in zip(doc_ids_info[i], doc_embeddings_info[i], batch_D_info[i]):
            doc_to_data[doc_id]["s2"] = s2
            if "emb" not in doc_to_data[doc_id]:
                doc_to_data[doc_id]["emb"] = emb

        merged_doc_ids = []
        merged_scores = []

        query_emb = query_embeddings[i]
        info_emb = info_embeddings[i]

        for doc_id, data in doc_to_data.items():
            emb = data["emb"]
            s1 = data.get("s1")
            if s1 is None:
                s1 = np.dot(emb, query_emb)
            s2 = data.get("s2")
            if s2 is None:
                s2 = np.dot(emb, info_emb)

            s1 = np.clip(s1, -1, 1)
            s2 = np.clip(s2, -1, 1)

            if consider_adaptive:
                s0 = np.dot(query_embeddings[i], info_embeddings[i])
                s0 = np.arccos(s0)
                s1 = np.arccos(s1)
                s2 = np.arccos(s2)
                alpha = 0.05848 * s0 + 0.45520
                score = -(alpha * s1 + (1 - alpha) * s2)
            else:
                score = s1 + s2

            merged_doc_ids.append(doc_id)
            merged_scores.append(score)

        topk_indices = np.argsort(-np.array(merged_scores))[:topk_new]
        selected_doc_ids.append([merged_doc_ids[idx] for idx in topk_indices])
        selected_scores.append([merged_scores[idx] for idx in topk_indices])

    return selected_doc_ids, np.array(selected_scores)
