"""
RAG (Retrieval-Augmented Generation) service for SmartLearn.

Uses sentence-transformers for embeddings and FAISS for vector search.
Supports three chunking modes: paragraph, character, character_overlap.
"""

from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────────

CHUNK_SIZE = 500       # characters per chunk (default)
CHUNK_OVERLAP = 100    # overlap between adjacent chunks (default)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cache the model so it loads once per process
_model: SentenceTransformer | None = None
_model_name: str | None = None


# ── Model Helpers ────────────────────────────────────────────────────────────

def model_tag(model_name: str) -> str:
    """Turn a model name into a safe filename suffix.

    >>> model_tag("sentence-transformers/all-MiniLM-L6-v2")
    'all_MiniLM_L6_v2'
    """
    # Take the last segment after any slash, then replace hyphens with underscores
    base = model_name.rsplit("/", 1)[-1]
    return base.replace("-", "_")


def resolve_model_source(model_name: str, artifact_root: str | Path | None = None) -> str:
    """Prefer a local cached model folder when it already exists.

    Checks (in order):
    1. artifact_root / hf_models / <model_tag>  (e.g. Day3/artifacts/hf_models/all-MiniLM-L6-v2)
    2. backend / artifacts / rag / hf_models / <model_tag>
    3. Fall back to the original model_name string (Hugging Face hub)
    """
    tag = model_tag(model_name)
    candidates: list[Path] = []

    if artifact_root is not None:
        candidates.append(Path(artifact_root) / "hf_models" / tag)

    # Also check the backend's own artifact directory
    backend_artifacts = Path(__file__).resolve().parent.parent / "artifacts" / "rag" / "hf_models" / tag
    candidates.append(backend_artifacts)

    for candidate in candidates:
        if candidate.exists() and (candidate / "config_sentence_transformers.json").exists():
            return str(candidate)

    return model_name


def get_device() -> str:
    """Choose CPU or CUDA for the current machine. CPU-first."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_model(model_name: str, device: str | None = None) -> SentenceTransformer:
    """Create or reuse one sentence-transformer model instance.

    Caches the model globally so repeated calls return the same instance.
    """
    global _model, _model_name

    if device is None:
        device = get_device()

    cache_key = f"{model_name}@{device}"

    if _model is not None and _model_name == cache_key:
        return _model

    source = resolve_model_source(model_name)
    _model = SentenceTransformer(
        source,
        device=device,
        model_kwargs={"use_safetensors": False} if Path(source).exists() else {},
    )
    _model_name = cache_key
    return _model


def _get_model() -> SentenceTransformer:
    """Lazy-load the default embedding model.  Kept for backward compatibility."""
    return load_model(EMBEDDING_MODEL)


# ── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize one extracted page of PDF text.

    Removes null bytes, soft hyphens, repeated whitespace, and noisy line breaks.
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove soft hyphens (U+00AD)
    text = text.replace("­", "")

    # Collapse repeated whitespace (spaces, tabs, newlines) into single spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ newlines into at most 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ── PDF Parsing ──────────────────────────────────────────────────────────────

def extract_pages_for_rag(pdf_path: str | Path) -> list[dict]:
    """Read a PDF page by page, clean the text, and return [{page, text}] records.

    - Keeps original PDF page numbers (1-indexed).
    - Removes empty extracted text blocks.
    - Does not hard-code a page limit.
    """
    reader = PdfReader(str(pdf_path))
    pages: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        if text:  # skip empty pages
            pages.append({"page": page_number, "text": text})
    return pages


def parse_pdf(pdf_path: str | Path) -> list[dict]:
    """Parse a PDF file and return a list of page dicts with text.

    Kept for backward compatibility with Day 2 code.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append({"page": page_number, "text": text})
    return pages


def parse_pdf_bytes(pdf_bytes: bytes) -> list[dict]:
    """Parse PDF from bytes and return a list of page dicts with text."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append({"page": page_number, "text": text})
    return pages


# ── JSON Helpers ─────────────────────────────────────────────────────────────

def save_json(data: object, path: str | Path) -> None:
    """Save one Python object to a UTF-8 JSON file. Creates parent folders when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: str | Path) -> object:
    """Read one saved JSON artifact back into Python."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preview_records(records: list[dict], columns: list[str] | None = None, rows: int = 5):
    """Show a small table for chosen columns so we can inspect page/chunk artifacts quickly.

    Returns a pandas DataFrame.  If columns is None or empty, shows all columns.
    """
    import pandas as pd

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    if columns:
        usable = [c for c in columns if c in frame.columns]
        return frame[usable].head(rows)
    return frame.head(rows)


# ── Chunking ─────────────────────────────────────────────────────────────────

def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """Split a single oversized text block into smaller pieces.

    Prefers natural boundaries (spaces, newlines) and avoids splitting
    in the middle of words whenever possible.
    """
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces: list[str] = []
    remaining = text

    while len(remaining) > chunk_size:
        # Try to split at the last space or newline within chunk_size
        search_region = remaining[:chunk_size]
        # Find best split point: prefer double-newline, then single newline, then space
        split_at = -1

        for sep in ["\n\n", "\n", " "]:
            pos = search_region.rfind(sep)
            if pos > chunk_size * 0.5:  # only use if it's not too early
                split_at = pos + len(sep)
                break

        if split_at == -1:
            # No natural boundary found — split exactly at chunk_size
            split_at = chunk_size

        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    if remaining:
        pieces.append(remaining)

    return pieces


def chunk_by_paragraph(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = 0,  # unused for paragraph mode, accepted for uniform signature
) -> list[dict]:
    """Convert paragraph-level records into chunks while preserving page numbers and order.

    Paragraph boundaries (double-newlines) are preserved as much as possible.
    When a single paragraph exceeds chunk_size, it is split with slice_long_text.
    """
    chunks: list[dict] = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        if not text:
            continue

        # Split into paragraphs at double-newline boundaries
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            if len(para) <= chunk_size:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": para,
                    "chunk_mode": "paragraph",
                })
                chunk_id += 1
            else:
                # Oversized paragraph — split into smaller pieces
                pieces = slice_long_text(para, chunk_size)
                for piece in pieces:
                    if piece:
                        chunks.append({
                            "chunk_id": chunk_id,
                            "page": page["page"],
                            "text": piece,
                            "chunk_mode": "paragraph",
                        })
                        chunk_id += 1

    return chunks


def chunk_by_characters(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = 0,
    chunk_mode: str = "character",
) -> list[dict]:
    """Create fixed-size sliding-window chunks.

    When overlap=0, plain fixed-size windows with no overlap (character mode).
    When overlap>0, overlapping windows (character_overlap mode).
    """
    chunks: list[dict] = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": chunk_text,
                    "chunk_mode": chunk_mode,
                })
                chunk_id += 1
            # Advance: overlap>0 means character_overlap mode
            step = max(1, chunk_size - overlap)
            start += step

    return chunks


def build_chunks(
    records: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Select the requested chunking strategy and return a uniform chunk schema.

    chunk_mode must be one of:
    - "paragraph"       — preserve paragraph boundaries
    - "character"        — fixed-size windows, no overlap
    - "character_overlap" — fixed-size windows with overlap
    - "langchain_recursive" — use LangChain RecursiveCharacterTextSplitter (Appendix A)

    Every chunk contains: chunk_id, page, text, chunk_mode.
    """
    if chunk_mode == "paragraph":
        return chunk_by_paragraph(records, chunk_size=chunk_size, overlap=overlap)

    elif chunk_mode == "character":
        return chunk_by_characters(records, chunk_size=chunk_size, overlap=0, chunk_mode="character")

    elif chunk_mode == "character_overlap":
        return chunk_by_characters(records, chunk_size=chunk_size, overlap=overlap, chunk_mode="character_overlap")

    elif chunk_mode == "langchain_recursive":
        return chunk_with_langchain_recursive(records, chunk_size=chunk_size, chunk_overlap=overlap)

    else:
        raise ValueError(
            f"Unknown chunk_mode: {chunk_mode!r}. "
            f"Expected one of: paragraph, character, character_overlap, langchain_recursive."
        )


# Keep the old function for backward compatibility
def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split pages into overlapping chunks.  Kept for backward compatibility."""
    return chunk_by_characters(pages, chunk_size=chunk_size, overlap=chunk_overlap, chunk_mode="character_overlap")


# ── Embedding ────────────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: SentenceTransformer | None = None,
    model_name: str | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Encode a list of texts into normalized float32 vectors.

    If model is None, loads the model via load_model(model_name or EMBEDDING_MODEL).
    Returns a 2-D numpy array of shape (len(texts), embedding_dim).
    """
    if model is None:
        name = model_name or EMBEDDING_MODEL
        model = load_model(name, device=device)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


# ── Artifact Paths ───────────────────────────────────────────────────────────

def artifact_paths_for(
    document_id: str,
    chunk_mode: str,
    model_name: str,
    artifact_root: str | Path,
) -> dict:
    """Decide where pages, chunks, embeddings, manifests, and indexes should be saved.

    Returns a dict with keys: raw_pages_path, chunk_path, embedding_path, manifest_path.
    """
    root = Path(artifact_root)
    tag = model_tag(model_name)

    return {
        "raw_pages_path": root / "raw_pages" / f"{document_id}_pages.json",
        "chunk_path": root / "chunks" / f"{document_id}_{chunk_mode}.json",
        "embedding_path": root / "embeddings" / f"{document_id}_{chunk_mode}_{tag}.npy",
        "manifest_path": root / "embeddings" / f"{document_id}_{chunk_mode}_{tag}.manifest.json",
    }


# ── Full Pipeline ────────────────────────────────────────────────────────────

def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    model_name: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """Build or reuse the full pages -> chunks -> embeddings -> manifest bundle.

    Saves:
    - raw pages JSON
    - chunk metadata JSON
    - embedding .npy file
    - manifest JSON

    Reuses saved outputs when the signature still matches.

    Returns a dict with keys: manifest, chunks, embeddings, pages.
    """
    if model_name is None:
        model_name = EMBEDDING_MODEL

    if artifact_root is None:
        artifact_root = Path(__file__).resolve().parent.parent / "artifacts" / "rag"

    device = get_device()
    paths = artifact_paths_for(document_id, chunk_mode, model_name, artifact_root)

    # --- Build the expected manifest ---
    expected_manifest = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "embedding_dim": 384,  # MiniLM-L6-v2 always returns 384-d vectors
        "device": device,
        "chunk_path": str(paths["chunk_path"]),
        "embedding_path": str(paths["embedding_path"]),
        "raw_pages_path": str(paths["raw_pages_path"]),
    }

    # --- Check cache: if manifest already matches, reuse ---
    manifest_path = paths["manifest_path"]
    if manifest_path.exists():
        try:
            cached_manifest = load_json(manifest_path)
            # Compare key fields (ignore path strings that may differ by machine)
            cacheable_keys = [
                "document_id", "pdf_name", "num_pages", "chunk_mode",
                "chunk_size", "overlap", "model_name", "embedding_dim",
            ]
            match = all(
                cached_manifest.get(k) == expected_manifest.get(k)
                for k in cacheable_keys
            )
            if match and Path(paths["embedding_path"]).exists() and Path(paths["chunk_path"]).exists():
                embeddings = np.load(paths["embedding_path"])
                chunks_data = load_json(paths["chunk_path"])
                return {
                    "manifest": cached_manifest,
                    "chunks": chunks_data,
                    "embeddings": embeddings,
                    "pages": pages,
                }
        except Exception:
            pass  # Cache miss — rebuild

    # --- Save raw pages ---
    save_json(pages, paths["raw_pages_path"])

    # --- Build chunks ---
    chunks = build_chunks(pages, chunk_mode=chunk_mode, chunk_size=chunk_size, overlap=overlap)

    # Update manifest with actual values
    expected_manifest["num_chunks"] = len(chunks)

    # --- Save chunks ---
    save_json(chunks, paths["chunk_path"])

    # --- Generate embeddings ---
    model = load_model(model_name, device=device)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model=model, batch_size=batch_size)

    expected_manifest["embedding_dim"] = int(embeddings.shape[1])

    # --- Save embeddings ---
    Path(paths["embedding_path"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["embedding_path"], embeddings)

    # --- Save manifest ---
    save_json(expected_manifest, manifest_path)

    return {
        "manifest": expected_manifest,
        "chunks": chunks,
        "embeddings": embeddings,
        "pages": pages,
    }


# ── FAISS Index & Search ─────────────────────────────────────────────────────

def build_index(chunks: list[dict]) -> tuple[faiss.Index, np.ndarray]:
    """Build a FAISS index from chunk texts.

    Returns (index, embeddings_array).
    """
    model = _get_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    # FAISS expects float32
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance
    index.add(embeddings)

    return index, embeddings


def search(
    query: str,
    chunks: list[dict],
    index: faiss.Index,
    top_k: int = 5,
) -> list[dict]:
    """Search the FAISS index for chunks relevant to the query.

    Returns the top-k chunks with their similarity scores added.
    """
    model = _get_model()
    query_embedding = model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(1.0 / (1.0 + dist))  # convert L2 distance to similarity
        results.append(chunk)

    return results


# ── Appendix A: LangChain RecursiveCharacterTextSplitter ──────────────────────

def chunk_with_langchain_recursive(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[dict]:
    """Split pages using LangChain's RecursiveCharacterTextSplitter.

    Uses separator priority: double-newline → single newline → space → character fallback.
    Keeps the same chunk record format as the rest of rag.py.

    Raises ImportError if langchain-text-splitters is not installed.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "langchain-text-splitters is required for langchain_recursive mode. "
            "Install it with: pip install langchain-text-splitters"
        )

    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
    )

    chunks: list[dict] = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        if not text:
            continue

        page_chunks = splitter.split_text(text)
        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page["page"],
                    "text": chunk_text,
                    "chunk_mode": "langchain_recursive",
                })
                chunk_id += 1

    return chunks
