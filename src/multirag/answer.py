from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Optional
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def looks_like_citations(text: str) -> bool:
    t = text.strip()
    if len(t) < 80:
        return True
    if re.search(r"\bdoi:\b", t.lower()) and re.search(r"\(\d+\)", t):
        return True
    if t.count(";") > 8 and t.count(".") < 3:
        return True
    return False

def boost_by_keywords(hits, keywords: Sequence[str]):
    boosted = []
    for h in hits:
        t = h.node.text.lower()
        bonus = sum(1 for k in keywords if k in t)
        boosted.append((h, bonus))
    boosted.sort(key=lambda x: (x[1], x[0].score), reverse=True)
    return [h for h, _ in boosted]

DEFAULT_GUIDELINE_KEYWORDS = [
    "recommend", "recommendation", "strong recommendation", "conditional recommendation",
    "target blood pressure", "initiation", "threshold", "first-line", "treatment goal",
    "<140/90", "<130", "mmhg", "algorithm", "implementation"
]

DEFAULT_LLM_KEYWORDS = [
    "speculative decoding", "draft model", "verification", "acceptance rate",
    "speedup", "throughput", "latency", "tokens per second", "kv cache",
    "prefill", "decode", "batch", "overhead"
]

@dataclass
class RagGenerator:
    model_name: str = "google/flan-t5-large"
    max_input_tokens: int = 1024
    reserved_for_instructions: int = 200
    device: str = "cuda"

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    @property
    def ctx_budget(self) -> int:
        return max(256, self.max_input_tokens - self.reserved_for_instructions)

    def pack_context_to_budget(self, chunks: List[str]) -> str:
        packed = []
        used = 0
        for ch in chunks:
            ch_tokens = len(self.tokenizer.encode(ch, add_special_tokens=False))
            if used + ch_tokens + 2 > self.ctx_budget:
                if not packed:
                    ids = self.tokenizer.encode(ch, add_special_tokens=False)[: self.ctx_budget]
                    return self.tokenizer.decode(ids)
                break
            packed.append(ch)
            used += ch_tokens + 2
        return "\n\n---\n\n".join(packed)

    def answer_from_hits(
        self,
        question: str,
        hits,
        *,
        booster: Optional[Callable] = None,
        filter_citations: bool = True,
        max_new_tokens: int = 200,
        keep_k: int = 10,
    ) -> str:
        useful = hits
        if filter_citations:
            useful = [h for h in hits if not looks_like_citations(h.node.text)] or hits

        if booster is not None:
            useful = booster(useful)
        useful = useful[:keep_k]

        raw_chunks = [h.node.text for h in useful]
        context = self.pack_context_to_budget(raw_chunks)

        prompt = (
            "Answer the question using ONLY the context. "
            "If the context is insufficient, say you don't know.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        return self.tokenizer.decode(out[0], skip_special_tokens=True)
