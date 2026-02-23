from curses import meta
import os
import requests
from multirag.routing import SemanticExampleRouter
from multirag.routing import RagDescriptionRouter

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8000")

def main():
    active = requests.get(f"{REGISTRY_URL}/rags/active", timeout=10).json()
    print("Loaded RAGs:", [r["rag_id"] for r in active])

    router = RagDescriptionRouter({r["rag_id"]: r["description"] for r in active})

    tests = [
        "What target blood pressure does WHO recommend for hypertension without comorbidities?",
        "What affects speedup in speculative decoding? Explain acceptance rate and verification overhead."
    ]

    for q in tests:
        rid, meta = router.route(q)
        score = None
        if isinstance(meta, dict):
            score = meta.get("best_sim")
        else:
            score = meta  # if you ever switch to float return

        print("\nQ:", q)
        print("Routed to:", rid, "score:", None if score is None else round(float(score), 4))
        print("Meta:", meta)

if __name__ == "__main__":
    main()