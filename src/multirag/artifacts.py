from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss


def save_faiss_artifacts(
    out_dir: str,
    *,
    faiss_index: faiss.Index,
    nodes: List[Any],
    meta: Dict[str, Any],
) -> str:
    """
    Save a minimal, framework-agnostic artifact bundle:
      - faiss.index: vector index
      - chunks.jsonl: chunk text + metadata (vector_id aligned with index order)
      - meta.json: build config + compatibility info

    Returns: out_dir
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Save FAISS index
    faiss.write_index(faiss_index, str(out / "faiss.index"))

      # 2) Save chunks (aligned with vector ids)
    chunks_path = out / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for i, n in enumerate(nodes):
            text = n.get_content() if hasattr(n, "get_content") else str(n)

            md = dict(getattr(n, "metadata", None) or {})

            # --- normalize common filename keys -> "source" ---
            src = (
                md.get("source")
                or md.get("file_name")
                or md.get("filename")
                or md.get("file_path")
                or md.get("document_id")
            )
            if src is not None:
                md["source"] = str(src)
            # ---------------------------------------------------

            rec = {"vector_id": i, "text": text, "metadata": md}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 3) Save meta
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return str(out)


def load_faiss_artifacts(index_dir: str) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads:
      - faiss.index
      - chunks.jsonl
      - meta.json
    """
    d = Path(index_dir)
    faiss_index = faiss.read_index(str(d / "faiss.index"))

    chunks: List[Dict[str, Any]] = []
    with (d / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    with (d / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    return faiss_index, chunks, meta