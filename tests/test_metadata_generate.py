import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from multirag.metadata_generate import generate_rag_metadata


def _write_chunks(base: Path) -> Path:
    index_dir = base / "artifacts" / "finance" / "v1"
    index_dir.mkdir(parents=True, exist_ok=True)
    chunks = [
        {
            "vector_id": 0,
            "text": "Bond duration measures sensitivity to interest rate changes. Coupon bonds with longer maturity often have higher duration.",
            "metadata": {"source": "finance_book.pdf"},
        },
        {
            "vector_id": 1,
            "text": "Yield curve shape can signal inflation expectations, growth outlook, and monetary policy stance.",
            "metadata": {"source": "market_notes.pdf"},
        },
        {
            "vector_id": 2,
            "text": "Portfolio diversification reduces idiosyncratic risk while preserving expected return targets.",
            "metadata": {"source": "finance_book.pdf"},
        },
    ]
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for rec in chunks:
            f.write(json.dumps(rec) + "\n")
    return index_dir


def test_generate_rag_metadata_from_chunks_is_non_empty_and_deterministic(tmp_path):
    index_dir = _write_chunks(tmp_path)

    desc1, ex1 = generate_rag_metadata(
        rag_id="finance",
        label="Finance RAG",
        domain="finance",
        tags=["bonds", "portfolio"],
        data_uri=None,
        index_uri=str(index_dir),
        config={"chunk_size": 512, "chunk_overlap": 80},
        repo_root=tmp_path,
        max_examples=6,
    )
    desc2, ex2 = generate_rag_metadata(
        rag_id="finance",
        label="Finance RAG",
        domain="finance",
        tags=["bonds", "portfolio"],
        data_uri=None,
        index_uri=str(index_dir),
        config={"chunk_size": 512, "chunk_overlap": 80},
        repo_root=tmp_path,
        max_examples=6,
    )

    assert desc1
    assert ex1
    assert len(ex1) <= 6
    assert len({x.lower() for x in ex1}) == len(ex1)
    assert all(len(x) <= 120 for x in ex1)

    assert desc1 == desc2
    assert ex1 == ex2
