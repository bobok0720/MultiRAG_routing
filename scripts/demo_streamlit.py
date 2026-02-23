import json
import os
from typing import Any, Dict, List

import requests
import streamlit as st

from multirag.route_select import select_rag_with_fallback
from multirag.routing import SemanticExampleRouter
from multirag.runtime import RagInstanceManager


st.set_page_config(page_title="Multi-RAG Router Demo", page_icon=":mag:", layout="wide")


@st.cache_data(ttl=15)
def load_active_rags(registry_url: str) -> List[Dict[str, Any]]:
    r = requests.get(f"{registry_url}/rags/active", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_resource
def get_manager(repo_root: str) -> RagInstanceManager:
    return RagInstanceManager(repo_root=repo_root)


@st.cache_resource
def get_router(route_examples_key: str) -> SemanticExampleRouter:
    payload = json.loads(route_examples_key)
    return SemanticExampleRouter(payload)


def _router_key(active: List[Dict[str, Any]]) -> str:
    mapping = {r["rag_id"]: r.get("route_examples", []) for r in active}
    return json.dumps(mapping, sort_keys=True)


def _lookup_rag(active: List[Dict[str, Any]], rag_id: str) -> Dict[str, Any]:
    for r in active:
        if r["rag_id"] == rag_id:
            return r
    raise KeyError(f"Unknown rag_id: {rag_id}")


st.title("Multi-RAG Route + Retrieve Demo")

with st.sidebar:
    st.header("Settings")
    registry_url = st.text_input("REGISTRY_URL", value=os.getenv("REGISTRY_URL", "http://localhost:8000"))
    repo_root = st.text_input("REPO_ROOT", value=os.getenv("REPO_ROOT", os.getcwd()))
    retrieval_top_k = st.slider("Retrieval Top-K", min_value=1, max_value=10, value=5, step=1)
    candidates_k = st.slider("Fallback candidates", min_value=2, max_value=5, value=2, step=1)
    refresh = st.button("Refresh Active RAGs")

if refresh:
    load_active_rags.clear()

try:
    active = load_active_rags(registry_url)
except Exception as e:
    st.error(f"Failed to load active RAGs from {registry_url}: {e}")
    st.stop()

if not active:
    st.warning("No active RAGs found. Register and activate at least one RAG first.")
    st.stop()

st.subheader("Active RAG Registry")
for r in active:
    with st.expander(f"{r['rag_id']} | {r['label']}", expanded=False):
        st.write(f"**Domain:** {r.get('domain') or '-'}")
        st.write(f"**Tags:** {', '.join(r.get('tags') or []) or '-'}")
        st.write(f"**Version:** {r.get('active_version')}")
        st.write(f"**Description:** {r.get('description') or '-'}")
        st.write("**Route Examples:**")
        for ex in r.get("route_examples", []):
            st.write(f"- {ex}")

sample_questions = []
for r in active:
    sample_questions.extend(r.get("route_examples", []))
sample_questions = sample_questions[:20]

if sample_questions:
    chosen = st.selectbox("Use sample question (optional)", options=[""] + sample_questions, index=0)
else:
    chosen = ""

question = st.text_input("Ask a question", value=chosen)
run = st.button("Route + Retrieve")

if run:
    if not question.strip():
        st.warning("Enter a question.")
        st.stop()

    try:
        router = get_router(_router_key(active))
        manager = get_manager(repo_root)
    except Exception as e:
        st.error(f"Failed to initialize router/runtime: {e}")
        st.stop()

    try:
        selected_rag_id, selection_meta = select_rag_with_fallback(
            query=question,
            router=router,
            active_rags=active,
            instance_manager=manager,
            top_k_candidates=candidates_k,
        )
        selected = _lookup_rag(active, selected_rag_id)

        runtime = manager.load(
            rag_id=selected["rag_id"],
            label=selected["label"],
            index_uri=selected["index_uri"],
            config_json=json.dumps(selected["config"], sort_keys=True),
        )
        hits = runtime.retriever.retrieve(question, top_k=retrieval_top_k)
    except Exception as e:
        st.error(f"Route/retrieve failed: {e}")
        st.stop()

    st.subheader("Routing Result")
    st.write(f"**Selected RAG:** `{selected_rag_id}` ({selected['label']})")
    st.json(selection_meta)

    if selection_meta.get("mode") == "semantic_plus_retrieval":
        picked = selection_meta.get("picked") or {}
        st.write(
            f"Fallback picked `{picked.get('rag_id')}` "
            f"(semantic_sim={picked.get('semantic_sim')}, retrieval_q={picked.get('retrieval_q')})"
        )

    st.subheader("Top Retrieved Chunks")
    if not hits:
        st.info("No retrieval hits.")
    for i, h in enumerate(hits, 1):
        src = h.metadata.get("source") or h.metadata.get("file") or h.metadata.get("filename") or "-"
        with st.container():
            st.markdown(f"**[{i}] score={h.score:.4f} source={src}**")
            st.write((h.text or "")[:500] + ("..." if len(h.text or "") > 500 else ""))
