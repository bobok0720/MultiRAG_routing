from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional, Any

import torch
from sentence_transformers import SentenceTransformer, util

@dataclass
class RagEntry:
    key: str
    label: str
    retriever: Any
    booster: Optional[Callable] = None

class SemanticExampleRouter:
    '''Route by comparing the query embedding to example utterances per RAG.'''
    def __init__(self, route_examples: Dict[str, List[str]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        keys = list(route_examples.keys())
        texts = [t for k in keys for t in route_examples[k]]
        owners = [k for k in keys for _ in route_examples[k]]
        self._owners = owners
        self._emb = self.encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)

    def route(self, query: str, sim_threshold: float = 0.35, margin: float = 0.03) -> Tuple[str, Dict[str, float]]:
        q_emb = self.encoder.encode([query], normalize_embeddings=True, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, self._emb)[0]
        best_idx = int(torch.argmax(sims).item())
        best_sim = float(sims[best_idx].item())
        best_key = self._owners[best_idx]
        top2 = torch.topk(sims, k=min(2, sims.numel())).values
        second_sim = float(top2[1].item()) if top2.numel() > 1 else -1.0
        confident = (best_sim >= sim_threshold) and ((best_sim - second_sim) >= margin)
        return best_key, {"best_sim": best_sim, "second_sim": second_sim, "confident": float(confident)}

def looks_like_refusal(ans: str) -> bool:
    a = ans.lower()
    return ("don't know" in a) or ("not enough information" in a) or ("insufficient" in a)
