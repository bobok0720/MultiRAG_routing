from __future__ import annotations
import urllib.request
from pathlib import Path
from typing import Optional

def download_pdf(url: str, out_dir: str, filename: Optional[str] = None) -> str:
    '''Download a PDF to out_dir. Returns the saved filepath.'''
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0] or "download.pdf"
        if not filename.endswith(".pdf"):
            filename += ".pdf"
    out_path = str(Path(out_dir) / filename)
    urllib.request.urlretrieve(url, out_path)
    return out_path

def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
