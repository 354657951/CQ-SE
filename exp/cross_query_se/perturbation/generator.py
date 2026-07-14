# Query perturbation generator using Qwen2.5-7B-Instruct via LEMMA MaaS API.
# Generates K meaning-preserving perturbations (paraphrases + perspective reframes) per query.

import os
import re
import time
import logging
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert at reformulating questions. Your task is to generate diverse, "
    "meaning-preserving question rewrites. Each rewrite must:\n"
    "1. Preserve the original question's answer (semantics intact)\n"
    "2. Vary the surface form — use different words, phrasings, or structural patterns\n"
    "3. Vary retrieval-relevant terms — use entity aliases, appositions, time rephrasing, "
    "or perspective shifts (e.g., ask about a related entity that implies the same answer)\n"
    "Do NOT change the question's factual intent or correct answer."
)

USER_TEMPLATE = (
    "Generate exactly {k} diverse rewrites of the following question.\n"
    "Mix surface paraphrases (different wording, same structure) with perspective reframes "
    "(entity aliases, appositions, relational rephrasing, time phrasing changes).\n\n"
    "Original question: {question}\n\n"
    "Output ONLY a numbered list (1. ... 2. ... etc.), one rewrite per line. "
    "No explanations, no preamble."
)


def _parse_perturbations(text: str, k: int) -> List[str]:
    lines = text.strip().split("\n")
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned and len(cleaned) > 5:
            results.append(cleaned)
    return results[:k]


class PerturbationGenerator:
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        k: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        max_retries: int = 3,
        max_workers: int = 20,
    ):
        base_url = os.environ.get("LEMMA_MAAS_BASE_URL", "")
        api_key = os.environ.get("LEMMA_MAAS_API_KEY", "")
        self._client = None
        self._base_url = f"http://{base_url}/v1"
        self._api_key = api_key
        self.model = model
        self.k = k
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_workers = max_workers

    @property
    def client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("LEMMA_MAAS_API_KEY not set; cannot call LLM for perturbation generation")
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def _generate_single(
        self, question: str, temperature: Optional[float] = None, k: Optional[int] = None
    ) -> List[str]:
        temp = temperature if temperature is not None else self.temperature
        num_k = k if k is not None else self.k
        prompt = USER_TEMPLATE.format(k=num_k, question=question)

        # Fast-fail if no API key
        if not self._api_key:
            return []

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temp,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                text = response.choices[0].message.content or ""
                perts = _parse_perturbations(text, num_k)
                if perts:
                    return perts
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for question '{question[:50]}': {e}")
                time.sleep(2 ** attempt)
        return []

    def generate(self, question: str, temperature: Optional[float] = None) -> List[str]:
        return self._generate_single(question, temperature=temperature)

    def generate_batch(
        self, questions: List[str], temperature: Optional[float] = None
    ) -> List[List[str]]:
        results = [None] * len(questions)

        def worker(idx: int, q: str):
            return idx, self._generate_single(q, temperature=temperature)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(worker, i, q): i for i, q in enumerate(questions)}
            for future in as_completed(futures):
                idx, perts = future.result()
                results[idx] = perts if perts is not None else []

        return results
