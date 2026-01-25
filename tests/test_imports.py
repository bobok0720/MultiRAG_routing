def test_imports():
    import multirag
    from multirag.ingestion import download_pdf
    from multirag.indexing import build_faiss_rag_index
    from multirag.answer import RagGenerator
    from multirag.routing import SemanticExampleRouter
