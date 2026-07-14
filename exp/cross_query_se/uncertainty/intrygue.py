# INTRYGUE-style induction-aware entropy gating for uncertainty estimation.
# Efficient implementation: uses single forward pass (logits only, no output_attentions)
# for entropy computation. Uses a lightweight SinkRate approximation via attention
# output hooks on selected layers only, with short max_seq_len to avoid OOM.
# Reference: "INTRYGUE: Induction-Aware Entropy Gating for Reliable RAG Uncertainty Estimation"
# https://arxiv.org/abs/2603.21607

import gc
import math
import logging
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _sink_rate_from_attn_row(
    attn_row: torch.Tensor,
    input_ids: torch.Tensor,
    t: int,
) -> float:
    """
    SinkRate at position t = sum_{j<t-1} attn[t, j+1] * (ids[j] == ids[t-1]).
    attn_row: [kv_len] attention weights for query position t.
    input_ids: [full_seq_len] all token ids.
    """
    if t < 2 or len(attn_row) < 2:
        return 0.0
    pred_token = input_ids[t - 1].item()
    kv_len = len(attn_row)
    # positions j = 0..min(t-2, kv_len-2): ids[j] == pred_token → attend to j+1
    ids_prefix = input_ids[:min(t - 1, kv_len - 1)]
    indicator = (ids_prefix == pred_token).float()
    attn_slice = attn_row[1:1 + len(indicator)]  # attn[j+1] for j=0..
    n = min(len(attn_slice), len(indicator))
    return float((attn_slice[:n] * indicator[:n]).sum().item())


def identify_induction_heads(
    model,
    tokenizer,
    calib_texts: List[str],
    top_k: int = 10,
    device: str = "cuda:0",
    max_length: int = 48,
) -> List[Tuple[int, int]]:
    """
    Identify top-k induction heads by average SinkRate across calibration prompts.
    Uses very short sequences (max_length=48) to avoid OOM with output_attentions=True.
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    sink_rate_accum = torch.zeros(num_layers, num_heads)
    count = 0

    model.eval()
    with torch.no_grad():
        for text_idx, text in enumerate(calib_texts):
            if text_idx % 5 == 0:
                logger.info(f"Calibration {text_idx}/{len(calib_texts)}")
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            input_ids = enc["input_ids"][0]
            seq_len = input_ids.shape[0]
            if seq_len < 4:
                continue

            try:
                outputs = model(**enc, output_attentions=True, use_cache=False)
            except RuntimeError as e:
                logger.warning(f"OOM at calibration {text_idx} (seq={seq_len}), skip: {e}")
                torch.cuda.empty_cache()
                continue

            input_ids_cpu = input_ids.cpu()
            for layer_idx, layer_attn in enumerate(outputs.attentions):
                # [1, num_heads, seq_len, seq_len]
                la = layer_attn[0].float().cpu()  # [num_heads, seq, seq]
                for head_idx in range(la.shape[0]):
                    srs = [
                        _sink_rate_from_attn_row(la[head_idx, t, :], input_ids_cpu, t)
                        for t in range(2, seq_len)
                    ]
                    sink_rate_accum[layer_idx, head_idx] += (
                        float(sum(srs) / len(srs)) if srs else 0.0
                    )

            del outputs
            torch.cuda.empty_cache()
            count += 1

    if count == 0:
        logger.warning("No calibration prompts processed; using default heads")
        return [(l, h) for l in range(2) for h in range(top_k // 2)][:top_k]

    sink_rate_accum /= count
    flat = sorted(
        [(float(sink_rate_accum[l, h]), l, h) for l in range(num_layers) for h in range(num_heads)],
        key=lambda x: -x[0],
    )
    top_heads = [(l, h) for _, l, h in flat[:top_k]]
    logger.info(
        f"Top-{top_k} induction heads ({count} calibration prompts): "
        + ", ".join(f"L{l}H{h}(sr={flat[i][0]:.4f})" for i, (l, h) in enumerate(top_heads))
    )
    return top_heads


class INTRYGUEScorer:
    """
    INTRYGUE gated uncertainty scorer.
    Uses forward hooks on selected layers to capture attention weights for induction heads.
    Runs a single forward pass over (prompt + pre-generated answer).
    max_seq_len controls truncation to avoid OOM.
    """

    def __init__(
        self,
        model,
        tokenizer,
        induction_heads: List[Tuple[int, int]],
        device: str = "cuda:0",
        max_seq_len: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.induction_heads = induction_heads
        self.device = device
        self.max_seq_len = max_seq_len

        self.heads_by_layer: Dict[int, List[int]] = {}
        for (l, h) in induction_heads:
            self.heads_by_layer.setdefault(l, []).append(h)

        # Per-layer attention cache filled by hooks
        self._attn_cache: Dict[int, torch.Tensor] = {}
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        layers = self.model.model.layers
        for layer_idx in self.heads_by_layer:
            if layer_idx >= len(layers):
                continue
            h = layers[layer_idx].self_attn.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(h)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, inputs, output):
            # Qwen2 eager attention with output_attentions=True:
            # output = (hidden_states, attn_weights, past_kv)
            # attn_weights: [batch, heads, q_len, kv_len]
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                # Only store on CPU to avoid GPU memory overhead
                self._attn_cache[layer_idx] = output[1].float().cpu()
        return hook_fn

    def _clear_cache(self):
        self._attn_cache.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def score_from_answer(self, prompt: str, answer: str) -> Dict[str, float]:
        """
        Compute INTRYGUE scores for a single (prompt, answer) pair.
        Single forward pass with output_attentions=True (hooks capture target layers only).
        """
        self._clear_cache()

        p_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        a_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        if not a_ids:
            return {"min_max": 0.0, "mean": 0.0}

        # Truncate to max_seq_len
        total = len(p_ids) + len(a_ids)
        if total > self.max_seq_len:
            keep_p = max(self.max_seq_len - len(a_ids), 16)
            p_ids = p_ids[-keep_p:]
        prompt_len = len(p_ids)
        full_ids = p_ids + a_ids
        seq_len = len(full_ids)
        ids_tensor = torch.tensor([full_ids], device=self.device)
        ids_cpu = torch.tensor(full_ids)

        try:
            out = self.model(
                input_ids=ids_tensor,
                output_attentions=True,  # triggers hooks + attn weight computation
                use_cache=False,
            )
        except RuntimeError as e:
            logger.warning(f"OOM (seq={seq_len}): {e}")
            torch.cuda.empty_cache()
            self._clear_cache()
            return {"min_max": 0.0, "mean": 0.0}

        # Entropy at each answer token position
        logits = out.logits[0].float().cpu()  # [seq_len, vocab]
        ans_len = len(a_ids)
        entropies = []
        for k in range(ans_len):
            lp = prompt_len - 1 + k  # logit predicting answer token k
            if lp >= seq_len:
                break
            log_p = F.log_softmax(logits[lp], dim=-1)
            p = log_p.exp()
            # Use -sum(p * log_p) but guard against 0 * -inf = nan
            H = float(torch.where(p > 0, -p * log_p, torch.zeros_like(p)).sum().item())
            if math.isnan(H) or math.isinf(H):
                H = 0.0
            entropies.append(H)

        # SinkRate per answer token
        sink_rates = []
        n_ans = min(len(entropies), ans_len)
        for k in range(n_ans):
            t = prompt_len + k  # position of answer token k in full sequence
            if t < 2:
                sink_rates.append(0.0)
                continue
            head_srs = []
            for layer_idx, head_list in self.heads_by_layer.items():
                if layer_idx not in self._attn_cache:
                    continue
                la = self._attn_cache[layer_idx]  # [1, heads, seq, seq] on CPU
                if la.shape[2] <= t or la.shape[3] < 2:
                    continue
                for head_idx in head_list:
                    if head_idx >= la.shape[1]:
                        continue
                    attn_row = la[0, head_idx, t, :]  # [seq_len]
                    sr = _sink_rate_from_attn_row(attn_row, ids_cpu, t)
                    head_srs.append(sr)
            sink_rates.append(float(sum(head_srs) / len(head_srs)) if head_srs else 0.0)

        del out, logits
        self._clear_cache()
        torch.cuda.empty_cache()

        if not entropies or not sink_rates:
            return {"min_max": 0.0, "mean": 0.0}

        n = min(len(entropies), len(sink_rates))
        e, s = entropies[:n], sink_rates[:n]
        return {
            "min_max": float(min(s)) * float(max(e)),
            "mean": float(sum(s) / n) * float(sum(e) / n),
        }

    def score_batch_from_answers(
        self, prompts: List[str], answers: List[str]
    ) -> List[Dict[str, float]]:
        results = []
        for i, (p, a) in enumerate(zip(prompts, answers)):
            if i % 200 == 0:
                logger.info(f"INTRYGUE scoring: {i}/{len(prompts)}")
            results.append(self.score_from_answer(p, a))
        return results
