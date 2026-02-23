import requests

BASE = "http://localhost:8000"

payloads = [
    {
        "rag_id": "health",
        "label": "WHO Hypertension RAG",
        "owner": "bobok",
        "domain": "health",
        "tags": ["guideline", "hypertension"],
        "config": {
            "index_type": "faiss_llamaindex",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "faiss_dim": 384,
            "chunk_size": 512,
            "chunk_overlap": 80,
            "similarity_top_k": 12,
            "boost_keywords": ["recommend", "target blood pressure", "mmhg"],
        },
        # description/route_examples intentionally omitted
        # to exercise server-side metadata generation.
        "data_uri": "data/pdfs_health",
        "index_uri": "artifacts/health/v1",
    },
    {
        "rag_id": "llm",
        "label": "Speculative Decoding (NAACL 2025) RAG",
        "owner": "bobok",
        "domain": "llm_systems",
        "tags": ["speculative-decoding", "inference"],
        "config": {
            "index_type": "faiss_llamaindex",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "faiss_dim": 384,
            "chunk_size": 512,
            "chunk_overlap": 80,
            "similarity_top_k": 12,
            "boost_keywords": ["speculative decoding", "acceptance rate", "speedup", "throughput"],
        },
        # description/route_examples intentionally omitted
        # to exercise server-side metadata generation.
        "data_uri": "data/pdfs_llm",
        "index_uri": "artifacts/llm/v1",
    },
]

for p in payloads:
    r = requests.post(f"{BASE}/rags/register", json=p)
    r.raise_for_status()
    reg = r.json()
    print("Registered:", reg)
    a = requests.post(f"{BASE}/rags/{p['rag_id']}/activate", params={"version": reg["version"]})
    a.raise_for_status()
    print("Activated:", a.json())

print("Active RAGs:", requests.get(f"{BASE}/rags/active").json())
