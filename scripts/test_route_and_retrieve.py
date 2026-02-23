import os, json, requests
from multirag.routing import SemanticExampleRouter
from multirag.runtime import RagInstanceManager

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8000")
REPO_ROOT = os.getenv("REPO_ROOT", os.getcwd())

def main():
    active = requests.get(f"{REGISTRY_URL}/rags/active", timeout=10).json()

    # route examples router (your current router)
    router = SemanticExampleRouter({r["rag_id"]: r["route_examples"] for r in active})

    mgr = RagInstanceManager(repo_root=REPO_ROOT)

    queries = [
        "What target blood pressure does WHO recommend for hypertension without comorbidities?",
        "What affects speedup in speculative decoding? Explain acceptance rate and verification overhead.",
        "What is bond duration and why does it matter for interest rate risk?",
        "Explain the Euler–Lagrange equation and what generalized coordinates mean.",
        "What is a load path in a building and why is it important?",
        "What is a ‘model’?"
    ]

    for q in queries:
        rid, meta = router.route(q)
        print("\nQ:", q)
        print("SEMANTIC ROUTE:", rid, "meta:", meta)

        # If not confident, evaluate top-2 by retrieval evidence
        if not bool(meta.get("confident")):
            candidates = router.route_topk(q, k=2)
            print("NOT CONFIDENT → trying candidates:", candidates)

            best = None
            best_hits = []

            for cand_id, sim in candidates:
                r = next(x for x in active if x["rag_id"] == cand_id)
                runtime = mgr.load(
                    rag_id=r["rag_id"],
                    label=r["label"],
                    index_uri=r["index_uri"],
                    config_json=json.dumps(r["config"]),
                )
                hits = runtime.retriever.retrieve(q, top_k=5)

                # retrieval quality heuristic: best hit score
                top_scores = [h.score for h in hits[:3]]
                quality = (sum(top_scores) / len(top_scores)) if top_scores else -1.0

                print(f"  candidate={cand_id} semantic_sim={sim:.4f} retrieval_best={quality:.4f}")

                            # Patch 2: safety if best never got set
            if best is None:
                # fallback to semantic winner
                rag_id = rid
                r = next(x for x in active if x["rag_id"] == rag_id)
                runtime = mgr.load(
                    rag_id=r["rag_id"],
                    label=r["label"],
                    index_uri=r["index_uri"],
                    config_json=json.dumps(r["config"]),
                )
                k = int(r["config"].get("similarity_top_k", 8))
                hits = runtime.retriever.retrieve(q, top_k=min(5, k))
                print("FINAL ROUTE (fallback to semantic):", rag_id)
            else:
                rag_id = best["rag_id"]
                hits = best_hits
                print("FINAL ROUTE (semantic+retrieval):", best)

        else:
            # confident: just use semantic winner
            rag_id = rid
            r = next(x for x in active if x["rag_id"] == rag_id)
            runtime = mgr.load(
                rag_id=r["rag_id"],
                label=r["label"],
                index_uri=r["index_uri"],
                config_json=json.dumps(r["config"]),
            )
            hits = runtime.retriever.retrieve(q, top_k=5)
            print("FINAL ROUTE (semantic only):", rag_id)

        print("TOP HITS:")
        for i, h in enumerate(hits, 1):
            src = h.metadata.get("source") or h.metadata.get("file") or h.metadata.get("filename")
            print(f"  [{i}] score={h.score:.4f} src={src}")
            print("      ", h.text[:180].replace("\n", " "), "...")
        
if __name__ == "__main__":
    main()