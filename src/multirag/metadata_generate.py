from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "which",
    "why",
    "with",
}

_BOILERPLATE_PATTERNS = (
    "all rights reserved",
    "this page intentionally left blank",
    "copyright",
    "isbn",
)


def _resolve_path(path_str: str | None, repo_root: str | Path | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p

    root = Path(repo_root) if repo_root else Path.cwd()
    candidate = root / path_str
    if candidate.exists():
        return candidate
    return None


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _is_good_chunk(text: str) -> bool:
    t = _clean_whitespace(text).lower()
    if len(t) < 60:
        return False
    return not any(pattern in t for pattern in _BOILERPLATE_PATTERNS)


def _load_from_artifacts(index_dir: Path) -> Tuple[List[str], List[str]]:
    chunks_path = index_dir / "chunks.jsonl"
    if not chunks_path.exists():
        return [], []

    texts: List[str] = []
    sources: List[str] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = str(rec.get("text", "")).strip()
            if not text:
                continue
            texts.append(text)

            md = rec.get("metadata") or {}
            src = md.get("source") or md.get("file_name") or md.get("filename") or md.get("file_path")
            if src:
                sources.append(Path(str(src)).name)
    return texts, sources


def _load_from_data_uri(data_dir: Path, config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    from multirag.indexing import load_docs_and_split

    chunk_size = int(config.get("chunk_size", 512))
    chunk_overlap = int(config.get("chunk_overlap", 80))
    _, nodes = load_docs_and_split(str(data_dir), chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    texts: List[str] = []
    sources: List[str] = []
    for n in nodes:
        text = n.get_content() if hasattr(n, "get_content") else str(n)
        if text and text.strip():
            texts.append(text)
        md = dict(getattr(n, "metadata", None) or {})
        src = md.get("source") or md.get("file_name") or md.get("filename") or md.get("file_path")
        if src:
            sources.append(Path(str(src)).name)
    return texts, sources


def _extract_topics(texts: Sequence[str], max_topics: int = 8) -> List[str]:
    term_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()

    for text in texts:
        tokens = _tokenize(text)
        if not tokens:
            continue
        term_counts.update(tokens)
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        bigram_counts.update(bigrams)

    # Prefer meaningful bigrams; then complement with high-signal terms.
    top_bigrams = sorted(
        ((k, v) for k, v in bigram_counts.items() if v >= 2),
        key=lambda x: (-x[1], x[0]),
    )
    top_terms = sorted(
        ((k, v) for k, v in term_counts.items() if v >= 3 and len(k) >= 4),
        key=lambda x: (-x[1], x[0]),
    )

    out: List[str] = []
    seen_tokens = set()

    for phrase, _ in top_bigrams:
        a, b = phrase.split(" ", 1)
        if a in seen_tokens and b in seen_tokens:
            continue
        out.append(phrase)
        seen_tokens.add(a)
        seen_tokens.add(b)
        if len(out) >= max_topics:
            return out

    for term, _ in top_terms:
        if term in seen_tokens:
            continue
        out.append(term)
        seen_tokens.add(term)
        if len(out) >= max_topics:
            break
    return out


def _first_or_fallback(items: Sequence[str], idx: int, fallback: str) -> str:
    return items[idx] if len(items) > idx else fallback


def _build_description(
    rag_id: str,
    label: str | None,
    domain: str | None,
    tags: Sequence[str] | None,
    topics: Sequence[str],
    sources: Sequence[str],
) -> str:
    title = (label or rag_id).strip()
    domain_text = domain or (tags[0] if tags else "general")
    t1 = _first_or_fallback(topics, 0, "core concepts")
    t2 = _first_or_fallback(topics, 1, "key terminology")
    t3 = _first_or_fallback(topics, 2, "practical guidance")

    srcs = sorted({s for s in sources if s})[:2]
    if srcs:
        return (
            f"{title} focuses on {domain_text} topics including {t1}, {t2}, and {t3}. "
            f"Primary sources include {', '.join(srcs)}."
        )
    return f"{title} focuses on {domain_text} topics including {t1}, {t2}, and {t3}."


def _build_examples(
    rag_id: str,
    label: str | None,
    domain: str | None,
    topics: Sequence[str],
    max_examples: int,
) -> List[str]:
    scope = domain or label or rag_id
    seeds = list(topics)[: max_examples * 2]
    if not seeds:
        seeds = ["main principles", "key definitions", "practical applications"]

    templates = [
        "What are the key ideas in {topic} for {scope}?",
        "Explain {topic} with a concise example.",
        "How does {topic} affect decisions in {scope}?",
        "When should I use {topic} in practice?",
        "What are common mistakes related to {topic}?",
        "Give a quick summary of {topic}.",
    ]

    out: List[str] = []
    seen = set()
    for i, topic in enumerate(seeds):
        tmpl = templates[i % len(templates)]
        q = _clean_whitespace(tmpl.format(topic=topic, scope=scope))
        if len(q) > 120:
            q = q[:117].rstrip() + "..."
        if not q.endswith("?"):
            q += "?"
        ql = q.lower()
        if ql in seen:
            continue
        seen.add(ql)
        out.append(q)
        if len(out) >= max_examples:
            break
    return out


def generate_rag_metadata(
    *,
    rag_id: str,
    label: str | None,
    domain: str | None,
    tags: Sequence[str] | None,
    data_uri: str | None,
    index_uri: str | None,
    config: Dict[str, Any],
    repo_root: str | Path | None = None,
    max_examples: int = 6,
) -> Tuple[str, List[str]]:
    """
    Generate a deterministic description and example route questions.
    Data source preference:
      1) index_uri/chunks.jsonl
      2) data_uri via PDF loading + chunking
    """
    texts: List[str] = []
    sources: List[str] = []

    index_dir = _resolve_path(index_uri, repo_root)
    if index_dir is not None:
        texts, sources = _load_from_artifacts(index_dir)

    if not texts:
        data_dir = _resolve_path(data_uri, repo_root)
        if data_dir is not None:
            texts, sources = _load_from_data_uri(data_dir, config)

    if not texts:
        raise ValueError(
            "Could not generate metadata: no readable chunks at index_uri and no readable data_uri."
        )

    filtered = [_clean_whitespace(t) for t in texts if _is_good_chunk(t)]
    if not filtered:
        filtered = [_clean_whitespace(t) for t in texts if len(_clean_whitespace(t)) >= 40]
    if not filtered:
        raise ValueError("Could not generate metadata: no usable text content found.")

    topics = _extract_topics(filtered)
    description = _build_description(rag_id, label, domain, tags or [], topics, sources)
    examples = _build_examples(rag_id, label, domain, topics, max_examples=max_examples)

    # Hard guarantees
    if not examples:
        examples = [f"What are the main concepts covered in {label or rag_id}?"]
    examples = examples[:max_examples]
    return description, examples
