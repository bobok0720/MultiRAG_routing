from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import faiss
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    from llama_index.core import SimpleDirectoryReader
except Exception:
    from llama_index.readers.file import SimpleDirectoryReader


@dataclass
class RagIndexBundle:
    index: VectorStoreIndex
    retriever: object
    docs: list
    nodes: list
    faiss_index: faiss.Index

def build_faiss_rag_index(
    pdf_dir: str,
    *,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    chunk_size: int = 512,
    chunk_overlap: int = 80,
    similarity_top_k: int = 8,
    faiss_dim: Optional[int] = None,
) -> RagIndexBundle:
    '''Build a FAISS-backed vector RAG index from a directory of PDFs.'''
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model, normalize=normalize)

    docs = SimpleDirectoryReader(pdf_dir).load_data()
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)

    if faiss_dim is None:
        if "all-MiniLM-L6-v2" in embedding_model:
            faiss_dim = 384
        elif "bge-small" in embedding_model:
            faiss_dim = 512
        else:
            raise ValueError("faiss_dim is required for this embedding model. Pass faiss_dim=<dim>.")

    faiss_index = faiss.IndexFlatIP(faiss_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    index = VectorStoreIndex(nodes, vector_store=vector_store)
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    return RagIndexBundle(index=index, retriever=retriever, docs=docs, nodes=nodes, faiss_index=faiss_index)
def load_docs_and_split(pdf_dir: str, chunk_size: int, chunk_overlap: int):
    try:
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter
    except Exception:
        from llama_index.core import SimpleDirectoryReader
        from llama_index.core.node_parser import SentenceSplitter

    docs = SimpleDirectoryReader(input_dir=pdf_dir).load_data()
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)
    return docs, nodes