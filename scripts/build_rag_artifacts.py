
import argparse
from pathlib import Path

import numpy as np
import faiss
from multirag.indexing import build_faiss_rag_index
from multirag.artifacts import save_faiss_artifacts

from sentence_transformers import SentenceTransformer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_id", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--pdf_dir", required=True)
    ap.add_argument("--out_root", default="artifacts")

    # keep params configurable (later you can pull these from DB config)
    ap.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--chunk_overlap", type=int, default=80)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--faiss_dim", type=int, default=384)

    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.rag_id / f"v{args.version}"

    bundle = build_faiss_rag_index(
        args.pdf_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        similarity_top_k=args.top_k,
        faiss_dim=args.faiss_dim,
    )

    meta = {
        "rag_id": args.rag_id,
        "version": args.version,
        "pdf_dir": args.pdf_dir,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
        "faiss_dim": args.faiss_dim,
    }

    # --- Step A: explicitly build FAISS from chunk embeddings (guarantees ntotal > 0) ---
    nodes = bundle.nodes
    texts = [
        (n.get_content() if hasattr(n, "get_content") else str(n))
        for n in nodes
    ]

    if len(texts) == 0:
        raise RuntimeError("No chunks/nodes were produced. Check PDF loading and chunking.")

    encoder = SentenceTransformer(args.embedding_model)
    emb = encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    if emb.ndim != 2 or emb.shape[1] != args.faiss_dim:
        raise RuntimeError(
            f"Embedding dim mismatch: got {emb.shape}, expected (*, {args.faiss_dim}). "
            f"Model={args.embedding_model}"
        )

    faiss_index = faiss.IndexFlatIP(args.faiss_dim)
    faiss_index.add(emb)

    if faiss_index.ntotal == 0:
        raise RuntimeError("FAISS index is empty after add(). Something went wrong during embedding.")

    # Save artifacts using the FAISS index we just populated
    saved = save_faiss_artifacts(
        str(out_dir),
        faiss_index=faiss_index,
        nodes=nodes,
        meta=meta,
    )
    # --- end Step A ---


if __name__ == "__main__":
    main()