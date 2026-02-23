from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from multirag.artifacts import load_faiss_artifacts


@dataclass
class Hit:
    score: float
    text: str
    metadata: Dict[str, Any]


class ArtifactFaissRetriever:
    """
    Loads FAISS + chunks from an artifact directory and retrieves top-k chunks for a query.
    """
    def __init__(self, index_dir: str, embedding_model: str, top_k: int = 8):
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.top_k = top_k

        self.faiss_index, self.chunks, self.meta = load_faiss_artifacts(index_dir)
        self.encoder = SentenceTransformer(embedding_model)

        # sanity: dimension check
        dim = self.faiss_index.d
        test_vec = self.encoder.encode(["test"], normalize_embeddings=True)
        if test_vec.shape[1] != dim:
            raise ValueError(
                f"Embedding dim mismatch: artifacts index dim={dim}, "
                f"embedder dim={test_vec.shape[1]} (model={embedding_model})"
            )

    def retrieve(self, query: str, top_k: int | None = None) -> List[Hit]:
        k = top_k or self.top_k
        q = self.encoder.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.faiss_index.search(q, k)

        out: List[Hit] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            rec = self.chunks[int(idx)]
            out.append(Hit(score=float(score), text=rec["text"], metadata=rec.get("metadata", {})))
        return out