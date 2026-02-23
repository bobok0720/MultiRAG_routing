from __future__ import annotations
from typing import Dict, Any, List, Tuple

def retrieval_quality(hits) -> float:
    # simple and effective: best similarity score
    return max((h.score for h in hits), default=-1.0)

def select_rag_with_fallback(
    query: str,
    router,
    active_rags: List[Dict[str, Any]],
    instance_manager,
    top_k_candidates: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    rid, meta = router.route(query)

    # if confident, just use it
    if bool(meta.get("confident")):
        return rid, {"mode": "semantic_only", "router_meta": meta}

    # otherwise evaluate top candidates by retrieval evidence
    candidates = router.route_topk(query, k=top_k_candidates)

    best = None
    for cand_id, sim in candidates:
        r = next(x for x in active_rags if x["rag_id"] == cand_id)
        runtime = instance_manager.load(
            rag_id=r["rag_id"],
            label=r["label"],
            index_uri=r["index_uri"],
            config_json=__import__("json").dumps(r["config"]),
        )
        hits = runtime.retriever.retrieve(query, top_k=5)
        q = retrieval_quality(hits)

        rec = {"rag_id": cand_id, "semantic_sim": sim, "retrieval_q": q}
        if best is None or rec["retrieval_q"] > best["retrieval_q"]:
            best = rec

    return best["rag_id"], {"mode": "semantic_plus_retrieval", "candidates": candidates, "picked": best}