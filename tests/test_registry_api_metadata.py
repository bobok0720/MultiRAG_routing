import importlib
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _make_chunks(tmp_path: Path, rel: str) -> str:
    d = tmp_path / rel
    d.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "vector_id": 0,
            "text": "Load path in a building transfers gravity and lateral forces through slabs, beams, columns, and foundations.",
            "metadata": {"source": "construction_guide.pdf"},
        },
        {
            "vector_id": 1,
            "text": "Shear walls, bracing, and diaphragms improve structural stability under wind and seismic loading.",
            "metadata": {"source": "construction_guide.pdf"},
        },
    ]
    with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rel


def _build_client(monkeypatch, tmp_path: Path) -> TestClient:
    db_path = tmp_path / "registry.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("REPO_ROOT", str(tmp_path))

    # Ensure fresh module import after env vars are set.
    for mod in list(sys.modules):
        if mod.startswith("services.registry_api.app."):
            del sys.modules[mod]

    main_mod = importlib.import_module("services.registry_api.app.main")
    return TestClient(main_mod.app)


def test_register_generates_metadata_if_missing(monkeypatch, tmp_path):
    rel_index = _make_chunks(tmp_path, "artifacts/construction/v1")
    client = _build_client(monkeypatch, tmp_path)

    payload = {
        "rag_id": "construction",
        "label": "Construction RAG",
        "owner": "bobuk",
        "domain": "construction",
        "tags": ["building", "structural"],
        "config": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "faiss_dim": 384,
            "chunk_size": 512,
            "chunk_overlap": 80,
            "similarity_top_k": 8,
        },
        "index_uri": rel_index,
        "data_uri": None,
    }

    reg = client.post("/rags/register", json=payload)
    assert reg.status_code == 200, reg.text
    assert reg.json()["version"] == 1

    act = client.post("/rags/construction/activate", params={"version": 1})
    assert act.status_code == 200, act.text

    active = client.get("/rags/active")
    assert active.status_code == 200, active.text
    rows = active.json()
    assert len(rows) == 1
    assert rows[0]["description"]
    assert rows[0]["route_examples"]


def test_register_preserves_user_metadata(monkeypatch, tmp_path):
    rel_index = _make_chunks(tmp_path, "artifacts/finance/v1")
    client = _build_client(monkeypatch, tmp_path)

    payload = {
        "rag_id": "finance",
        "label": "Finance RAG",
        "owner": "bobuk",
        "domain": "finance",
        "tags": ["bonds"],
        "config": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "faiss_dim": 384,
            "chunk_size": 512,
            "chunk_overlap": 80,
            "similarity_top_k": 8,
        },
        "index_uri": rel_index,
        "description": "Custom description.",
        "route_examples": ["What is bond duration?", "How does yield affect bond prices?"],
    }

    reg = client.post("/rags/register", json=payload)
    assert reg.status_code == 200, reg.text

    act = client.post("/rags/finance/activate", params={"version": 1})
    assert act.status_code == 200, act.text

    active = client.get("/rags/active")
    assert active.status_code == 200, active.text
    rows = active.json()
    assert len(rows) == 1
    assert rows[0]["description"] == "Custom description."
    assert rows[0]["route_examples"] == payload["route_examples"]
