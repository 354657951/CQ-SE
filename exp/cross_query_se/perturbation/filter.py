# Semantic-equivalence filter for query perturbations.
# Two-stage (default): (1) DeBERTa-v2-XLarge-MNLI bidirectional entailment, (2) BGE cosine similarity >= tau.
# Cosine-only mode: skip NLI stage, use BGE cosine similarity only (better for question paraphrases).

import os
import logging
from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NLI_MODEL_ID = "microsoft/deberta-v2-xlarge-mnli"
BGE_MODEL_ID = "BAAI/bge-large-en-v1.5"

ENTAILMENT_LABEL = "ENTAILMENT"


class SemanticEquivalenceFilter:
    def __init__(
        self,
        tau: float = 0.85,
        device: str = "cuda",
        nli_batch_size: int = 32,
        emb_batch_size: int = 128,
        cosine_only: bool = False,
    ):
        self.tau = tau
        self.device = device
        self.nli_batch_size = nli_batch_size
        self.emb_batch_size = emb_batch_size
        self.cosine_only = cosine_only

        hf_home = os.environ.get("HF_HOME", None)
        cache_dir = os.path.join(hf_home, "hub") if hf_home else None

        if not cosine_only:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"Loading NLI model: {NLI_MODEL_ID}")
            self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID, cache_dir=cache_dir)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                NLI_MODEL_ID, cache_dir=cache_dir
            ).to(device)
            self.nli_model.eval()
            self.nli_labels = self.nli_model.config.id2label
        else:
            logger.info("cosine_only=True: skipping NLI model load")
            self.nli_tokenizer = None
            self.nli_model = None
            self.nli_labels = None

        logger.info(f"Loading embedding model: {BGE_MODEL_ID}")
        self.embed_model = SentenceTransformer(BGE_MODEL_ID, cache_folder=cache_dir)

    def _nli_batch(self, premise_list: List[str], hypothesis_list: List[str]) -> List[str]:
        results = []
        for i in range(0, len(premise_list), self.nli_batch_size):
            prems = premise_list[i : i + self.nli_batch_size]
            hyps = hypothesis_list[i : i + self.nli_batch_size]
            enc = self.nli_tokenizer(
                prems,
                hyps,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.nli_model(**enc).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            for p in preds:
                results.append(self.nli_labels[p].upper())
        return results

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.embed_model.encode(
            texts,
            batch_size=self.emb_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=self.device,
        )

    def filter_perturbations(
        self, original: str, perturbations: List[str]
    ) -> Tuple[List[str], List[bool]]:
        if not perturbations:
            return [], []

        n = len(perturbations)

        # Embedding cosine similarity (always used)
        all_texts = [original] + perturbations
        embs = self._embed(all_texts)
        orig_emb = embs[0:1]
        pert_embs = embs[1:]
        cosines = (orig_emb * pert_embs).sum(axis=1)

        if self.cosine_only:
            passed = []
            mask = []
            for i, cos in enumerate(cosines):
                ok = float(cos) >= self.tau
                mask.append(ok)
                if ok:
                    passed.append(perturbations[i])
            return passed, mask

        # Stage 1: Bidirectional NLI (only when cosine_only=False)
        originals = [original] * n
        forward_labels = self._nli_batch(originals, perturbations)
        backward_labels = self._nli_batch(perturbations, originals)
        nli_pass = [
            (f == ENTAILMENT_LABEL and b == ENTAILMENT_LABEL)
            for f, b in zip(forward_labels, backward_labels)
        ]

        passed = []
        mask = []
        for i, (nli_ok, cos) in enumerate(zip(nli_pass, cosines)):
            ok = nli_ok and float(cos) >= self.tau
            mask.append(ok)
            if ok:
                passed.append(perturbations[i])

        return passed, mask

    def filter_with_regeneration(
        self,
        original: str,
        perturbations: List[str],
        generator,
        k: int = 5,
        extra_temps: List[float] = None,
    ) -> List[str]:
        if extra_temps is None:
            extra_temps = [0.9, 1.0]

        passed, _ = self.filter_perturbations(original, perturbations)

        for temp in extra_temps:
            if len(passed) >= k:
                break
            try:
                new_perts = generator.generate(original, temperature=temp)
                new_perts = [p for p in new_perts if p not in passed]
                extra_passed, _ = self.filter_perturbations(original, new_perts)
                passed.extend(extra_passed)
            except Exception:
                break

        return passed[:k]
