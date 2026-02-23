from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from functools import lru_cache
from pathlib import Path

from multirag.artifact_retriever import ArtifactFaissRetriever


@dataclass
class RagRuntime:
    rag_id: str
    label: str
    index_uri: str
    config: Dict[str, Any]
    retriever: ArtifactFaissRetriever


class RagInstanceManager:
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)

    @lru_cache(maxsize=16)  # keep last 16 loaded instances
    def load(self, rag_id: str, label: str, index_uri: str, config_json: str) -> RagRuntime:
        # config_json is stringified to make lru_cache hashable
        import json
        config = json.loads(config_json)

        index_dir = str(self.repo_root / index_uri)
        emb = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        top_k = int(config.get("similarity_top_k", 8))

        retriever = ArtifactFaissRetriever(index_dir=index_dir, embedding_model=emb, top_k=top_k)
        return RagRuntime(rag_id=rag_id, label=label, index_uri=index_uri, config=config, retriever=retriever)