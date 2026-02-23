# Multi-RAG Router (Colab-friendly)

This repo organizes your notebook prototype into a structured project:

- Build multiple RAG indices (one folder per domain)
- Register them in a registry
- Route queries using semantic example matching
- Fallback to multiple RAGs when routing is uncertain

## Repo layout

- `notebooks/` — your original prototype notebook
- `src/multirag/` — reusable modules (ingestion, indexing, answering, routing)
- `scripts/` — runnable demos
- `data/` — local PDFs (gitignored)

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

> Install PyTorch separately if needed, depending on your CUDA/CPU setup.

## Run demo

Put PDFs in:
- `data/pdfs_health/` (e.g., WHO guideline PDFs)
- `data/pdfs_llm/` (LLM niche PDFs)

Then:

```bash
python scripts/demo.py --download_llm_pdf --question "What affects speedup in speculative decoding?"
```

## Streamlit presentation demo

Run registry API, register/activate RAGs, then launch:

```bash
streamlit run scripts/demo_streamlit.py
```
