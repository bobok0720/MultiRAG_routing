import argparse
from multirag.ingestion import download_pdf, ensure_dir
from multirag.indexing import build_faiss_rag_index
from multirag.answer import RagGenerator, boost_by_keywords, DEFAULT_GUIDELINE_KEYWORDS, DEFAULT_LLM_KEYWORDS
from multirag.routing import RagEntry, SemanticExampleRouter, looks_like_refusal

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--health_dir", default="data/pdfs_health")
    p.add_argument("--llm_dir", default="data/pdfs_llm")
    p.add_argument("--download_llm_pdf", action="store_true")
    p.add_argument("--question", required=True)
    args = p.parse_args()

    ensure_dir(args.health_dir)
    ensure_dir(args.llm_dir)

    if args.download_llm_pdf:
        download_pdf("https://aclanthology.org/2025.naacl-long.328.pdf", args.llm_dir, "naacl2025_decoding_speculative_decoding.pdf")

    rag1 = build_faiss_rag_index(args.health_dir, similarity_top_k=12)
    rag2 = build_faiss_rag_index(args.llm_dir, similarity_top_k=12)

    registry = {
        "health": RagEntry("health", "WHO Hypertension RAG", rag1.retriever, booster=lambda hits: boost_by_keywords(hits, DEFAULT_GUIDELINE_KEYWORDS)),
        "llm": RagEntry("llm", "Speculative Decoding (NAACL 2025) RAG", rag2.retriever, booster=lambda hits: boost_by_keywords(hits, DEFAULT_LLM_KEYWORDS)),
    }

    router = SemanticExampleRouter({
        "health": [
            "WHO guideline pharmacological treatment of hypertension in adults",
            "target blood pressure <140/90 <130 mmHg recommendation",
            "first-line drug classes for hypertension guideline",
        ],
        "llm": [
            "speculative decoding speedup depends on acceptance rate",
            "draft model verification overhead in speculative decoding",
            "throughput tokens per second improvement speculative decoding",
        ],
    })

    gen = RagGenerator()

    choice, info = router.route(args.question)
    entry = registry[choice]
    hits = entry.retriever.retrieve(args.question)
    ans = gen.answer_from_hits(args.question, hits, booster=entry.booster)

    if (not bool(info["confident"])) or looks_like_refusal(ans):
        candidates = []
        for k, e in registry.items():
            hits_k = e.retriever.retrieve(args.question)
            ans_k = gen.answer_from_hits(args.question, hits_k, booster=e.booster)
            candidates.append((k, ans_k))
        # Prefer non-refusal answers
        candidates.sort(key=lambda x: looks_like_refusal(x[1]))
        choice, ans = candidates[0]

    print(f"\nRouted to: {registry[choice].label}")
    print(f"Routing meta: {info}")
    print("\nAnswer:\n")
    print(ans)

if __name__ == "__main__":
    main()
