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
    'sentence-transformers_all-MiniLM-L6-v2'
    """
    return model_name.replace("/", "_").replace("\\", "_")


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


def extract_pages_from_bytes_for_rag(pdf_bytes: bytes) -> list[dict]:
    """Parse uploaded PDF bytes into cleaned [{page, text}] records for the backend upload route."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages: list[dict] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        if text:
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


def relative_path_str(path: str | Path, base: str | Path) -> str:
    """Return a shorter display path relative to a base folder.

    >>> relative_path_str(Path("/a/b/c/file.txt"), Path("/a/b"))
    'c/file.txt'
    """
    try:
        return str(Path(path).relative_to(Path(base)))
    except ValueError:
        return str(path)


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
    model_cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Encode a list of texts into normalized float32 vectors.

    If model is None, loads the model via load_model(model_name or EMBEDDING_MODEL).
    model_cache_dir is accepted for Chroma appendix compatibility (ignored; model resolution
    is handled by resolve_model_source internally).
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


def _index_dir_for(
    document_id: str,
    chunk_mode: str,
    model_name: str,
    chunk_size: int,
    overlap: int,
    artifact_root: str | Path,
) -> Path:
    """Return the directory path for FAISS index storage."""
    tag = model_tag(model_name)
    return (
        Path(artifact_root)
        / document_id
        / f"{chunk_mode}_c{chunk_size}_o{overlap}_{tag}"
    )


def ensure_index(
    document_id: str,
    pdf_name: str,
    pages: list[dict] | None = None,
    pdf_path: str | Path | None = None,
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """Build or reuse the full pages -> chunks -> embeddings -> FAISS index bundle.

    Returns a dict with keys: chunks, embeddings, manifest, index, index_path, meta_path.
    """
    if artifact_root is None:
        artifact_root = Path(__file__).resolve().parent.parent / "artifacts" / "rag"

    # Resolve pages from pdf_path if not provided
    if pages is None:
        if pdf_path is None:
            raise ValueError("One of pages or pdf_path must be provided")
        pages = extract_pages_for_rag(pdf_path)

    # Step 1: ensure chunks + embeddings exist (reuse Lab A artifacts)
    bundle = ensure_artifacts(
        document_id=document_id,
        pdf_name=pdf_name,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    # Step 2: determine FAISS index path
    index_dir = _index_dir_for(document_id, chunk_mode, model_name, chunk_size, overlap, artifact_root)
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "index.meta.json"

    # Step 3: build or load index
    if index_path.exists() and meta_path.exists():
        try:
            meta = load_json(meta_path)
            if (
                meta.get("num_chunks") == len(bundle["chunks"])
                and meta.get("embedding_dim") == bundle["embeddings"].shape[1]
            ):
                index = load_faiss_index(index_path)
                return {
                    "chunks": bundle["chunks"],
                    "embeddings": bundle["embeddings"],
                    "manifest": bundle["manifest"],
                    "index": index,
                    "index_path": str(index_path),
                    "meta_path": str(meta_path),
                }
        except Exception:
            pass  # Cache miss — rebuild

    # Build fresh index
    index = build_faiss_index(bundle["embeddings"])
    save_faiss_index(index, index_path)

    # Save metadata
    meta = {
        "document_id": document_id,
        "num_chunks": len(bundle["chunks"]),
        "embedding_dim": int(bundle["embeddings"].shape[1]),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
    }
    save_json(meta, meta_path)

    return {
        "chunks": bundle["chunks"],
        "embeddings": bundle["embeddings"],
        "manifest": bundle["manifest"],
        "index": index,
        "index_path": str(index_path),
        "meta_path": str(meta_path),
    }


def prepare_rag_document(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """Build a server-ready document record with pages, chunks, index paths, and empty history.

    The returned dict can be stored in documents[document_id] for the Day 2 app shape.
    """
    if artifact_root is None:
        artifact_root = Path(__file__).resolve().parent.parent / "artifacts" / "rag"

    index_bundle = ensure_index(
        document_id=document_id,
        pdf_name=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    paths = artifact_paths_for(document_id, chunk_mode, model_name, artifact_root)
    model_source = resolve_model_source(model_name, artifact_root)

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": pages,
        "chunks": index_bundle["chunks"],
        "chunk_size": len(index_bundle["chunks"]),
        "embedding_dim": index_bundle["manifest"]["embedding_dim"],
        "model_name": model_name,
        "model_source": model_source,
        "artifacts": {
            "index": index_bundle["index_path"],
            "chunks": str(paths["chunk_path"]),
            "embeddings": str(paths["embedding_path"]),
            "manifest": str(paths["manifest_path"]),
        },
        "history": [],
        "config": {
            "chunk_mode": chunk_mode,
            "chunk_size": chunk_size,
            "overlap": overlap,
        },
    }


# ── FAISS Index & Search ─────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS inner-product index from normalized embedding vectors.

    Normalized embeddings + inner product = cosine similarity.
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine similarity for normalized vectors)
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, index_path: str | Path) -> None:
    """Write the binary .faiss file to disk.

    Uses Python I/O (not FAISS C++ file handling) to avoid CJK path issues on Windows.
    """
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = faiss.serialize_index(index)
    # serialize_index returns an ndarray; convert to raw bytes for writing
    path.write_bytes(data.tobytes())


def load_faiss_index(index_path: str | Path) -> faiss.Index:
    """Load a saved .faiss index back into memory.

    Uses Python I/O (not FAISS C++ file handling) to avoid CJK path issues on Windows.
    """
    data = Path(index_path).read_bytes()
    arr = np.frombuffer(data, dtype=np.uint8)
    return faiss.deserialize_index(arr)


def build_index(chunks: list[dict]) -> tuple[faiss.Index, np.ndarray]:
    """Build a FAISS index from chunk texts.

    Returns (index, embeddings_array).
    Kept for backward compatibility with Day 2 code.
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


# ── Retrieval Helpers ────────────────────────────────────────────────────────

def keyword_set(text: str) -> set[str]:
    """Extract lightweight lexical tokens for simple reranking.

    Lowercases, splits on non-alpha, and filters tokens shorter than 2 chars.
    """
    return {t.lower() for t in re.split(r"[^a-zA-Z0-9]+", text) if len(t) >= 2}


def search_bundle(
    question: str,
    bundle: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    batch_size: int = 1,
    history: list[dict] | None = None,
) -> list[dict]:
    """Search a FAISS index in memory and return top-k hits.

    Each hit carries: page, chunk_id, text, score, chunk_mode.
    A small lexical rerank reorders the candidate pool before final top_k selection.
    """
    chunks = bundle["chunks"]
    index = bundle["index"]
    model_name = bundle.get("manifest", {}).get("model_name", EMBEDDING_MODEL)

    # Embed the question
    model = load_model(model_name)
    q_embedding = embed_texts([question], model=model, batch_size=batch_size)

    # Retrieve candidate_pool candidates from FAISS
    actual_pool = min(candidate_pool, len(chunks))
    distances, indices = index.search(q_embedding, actual_pool)

    # Build candidate hits
    candidates: list[dict] = []
    q_tokens = keyword_set(question)

    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx].copy()
        # Inner-product score (higher = more similar for normalized vectors)
        chunk["score"] = float(dist)
        # Lexical overlap bonus
        chunk_tokens = keyword_set(chunk["text"])
        overlap = len(q_tokens & chunk_tokens)
        chunk["_lex_overlap"] = overlap
        candidates.append(chunk)

    # Rerank: lexical overlap as tiebreaker after FAISS score
    candidates.sort(key=lambda c: (c["score"], c["_lex_overlap"]), reverse=True)

    # Return top_k without internal fields
    for c in candidates[:top_k]:
        c.pop("_lex_overlap", None)
        # Keep only the fields a hit should carry
        keep_keys = {"page", "chunk_id", "text", "score", "chunk_mode"}
        to_remove = [k for k in c if k not in keep_keys]
        for k in to_remove:
            c.pop(k, None)

    return candidates[:top_k]


def search_document(
    question: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    history: list[dict] | None = None,
) -> list[dict]:
    """Load a saved FAISS index from a prepared document, then retrieve top-k hits.

    Each hit carries: page, chunk_id, text, score.
    """
    index = load_faiss_index(document["artifacts"]["index"])
    bundle = {
        "chunks": document["chunks"],
        "index": index,
        "manifest": {
            "model_name": document.get("model_name", EMBEDDING_MODEL),
        },
    }
    return search_bundle(question, bundle, top_k=top_k, candidate_pool=candidate_pool, history=history)


def split_sentences(text: str) -> list[str]:
    """Split retrieved chunk text into candidate answer sentences."""
    # Split on sentence boundaries: .!? followed by space or newline
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def best_sentence_answer(question: str, hits: list[dict]) -> str:
    """Extract one short local answer sentence from retrieved hits.

    Picks the sentence with the most keyword overlap with the question.
    Appends a page tag like '(p. 3)' when possible.
    """
    if not hits:
        return "No relevant evidence found."

    q_tokens = keyword_set(question)
    best_sentence = ""
    best_score = -1
    best_page = hits[0]["page"]

    for hit in hits:
        sentences = split_sentences(hit["text"])
        for sent in sentences:
            sent_tokens = keyword_set(sent)
            overlap = len(q_tokens & sent_tokens)
            # Prefer sentences with more keyword overlap, break ties by length (shorter is better)
            score = overlap * 100 - len(sent) * 0.01
            if score > best_score:
                best_score = score
                best_sentence = sent
                best_page = hit["page"]

    if not best_sentence:
        # Fallback: return the beginning of the first hit
        best_sentence = hits[0]["text"][:200].strip()
        best_page = hits[0]["page"]

    # Truncate long sentences and add page tag
    if len(best_sentence) > 300:
        best_sentence = best_sentence[:297] + "..."

    return f"{best_sentence} (p. {best_page})"


# ── Project-Facing Wrappers ──────────────────────────────────────────────────

def extract_citations(answer: str, hits: list[dict] | None = None) -> list[int]:
    """Extract numeric PDF page citations from an answer string.

    Parses [Page X] markers, then falls back to unique hit pages.
    """
    import re as _re
    cited = [int(m) for m in _re.findall(r"\[Page\s+(\d+)\]", answer)]
    if cited:
        return sorted(set(cited))
    if hits:
        return sorted({h["page"] for h in hits})
    return []


def build_sources(hits: list[dict]) -> list[dict]:
    """Build frontend-friendly source objects from retrieval hits."""
    return [
        {
            "page": h["page"],
            "chunk_id": h["chunk_id"],
            "score": round(h["score"], 4),
            "preview": h["text"][:120].replace("\n", " "),
        }
        for h in hits
    ]


def answer_document(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "openrouter/free",
    history: list[dict] | None = None,
) -> dict:
    """Answer one question using retrieval + optional LLM.

    Returns a dict with: answer, citations, sources.
    Falls back to local extraction if no API key is available.
    Pass history (previous turns) so the LLM can resolve context-dependent questions.
    """
    # Retrieve
    hits = search_document(question, document, top_k=top_k, candidate_pool=candidate_pool)

    # Try LLM answering if API key exists
    answer = _try_llm_answer(question, hits, answer_model, history=history)
    if answer is None:
        answer = best_sentence_answer(question, hits)

    citations = extract_citations(answer, hits)
    sources = build_sources(hits)

    return {
        "answer": answer,
        "citations": citations,
        "sources": sources,
    }


def build_grounded_user_prompt(
    question: str,
    hits: list[dict],
    history: list[dict] | None = None,
) -> str:
    """Build a grounded LLM prompt combining history, retrieved evidence, and the question."""
    parts: list[str] = []

    # Recent chat history
    if history:
        parts.append("## Recent conversation")
        for turn in history[-4:]:  # last 4 turns only
            role = turn.get("role", "user")
            content = turn.get("content", turn.get("question", ""))
            parts.append(f"{role}: {content}")
        parts.append("")

    # Retrieved evidence
    parts.append("## Retrieved evidence from the document")
    for h in hits:
        parts.append(f"### [Page {h['page']}]\n{h['text']}")
        parts.append("")

    # New question
    parts.append(f"## Current question\n{question}")
    parts.append("\nAnswer the current question using only the retrieved evidence above. Cite sources with [Page X].")

    return "\n".join(parts)


def answer_document_turn(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "openrouter/free",
) -> dict:
    """Answer one question and auto-append the turn to the document's in-memory history.

    Returns the answer result plus the updated history list.
    """
    result = answer_document(
        document, question,
        top_k=top_k, candidate_pool=candidate_pool, answer_model=answer_model,
        history=document.get("history", []),
    )
    history = append_history(document, question, result)
    result["history"] = history
    return result


def answer_chat_turn(
    document: dict,
    message: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "openrouter/free",
) -> dict:
    """Route-level chat handler: fresh retrieval + answer + in-memory history update.

    Returns {answer, citations, sources} for the frontend.
    """
    result = answer_document_turn(document, message, top_k=top_k, candidate_pool=candidate_pool, answer_model=answer_model)
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
    }


def build_upload_response(document: dict) -> dict:
    """Build the visible upload success JSON from a richer server-side record."""
    pages = document.get("pages", [])
    characters = sum(len(p.get("text", "")) for p in pages)
    return {
        "status": "ok",
        "filename": document.get("filename", ""),
        "pages": len(pages),
        "characters": characters,
    }


def prepare_rag_chat_record(
    chat_id: str,
    filename: str,
    pdf_bytes: bytes | None = None,
    pages: list[dict] | None = None,
    upload_root: str | Path | None = None,
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """Build an upload-time documents[chat_id] record with pages, chunks, RAG index, and empty history.

    Accepts either pdf_bytes (for the backend upload route) or pre-extracted pages (for notebook tests).
    """
    if artifact_root is None:
        artifact_root = Path(__file__).resolve().parent.parent / "artifacts" / "rag"

    # Resolve pages from pdf_bytes if not provided
    if pages is None:
        if pdf_bytes is None:
            raise ValueError("One of pdf_bytes or pages must be provided")
        pages = extract_pages_from_bytes_for_rag(pdf_bytes)

    # Save uploaded PDF to disk
    if upload_root is None:
        upload_root = Path(__file__).resolve().parent.parent / "uploads"
    upload_dir = Path(upload_root)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf_path = upload_dir / f"{chat_id}.pdf"
    if pdf_bytes is not None:
        saved_pdf_path.write_bytes(pdf_bytes)

    # Build RAG artifacts
    rag_doc = prepare_rag_document(
        document_id=chat_id,
        filename=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        chunk_size=chunk_size,
        overlap=overlap,
        model_name=model_name,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    return {
        "chat_id": chat_id,
        "filename": filename,
        "saved_pdf_path": str(saved_pdf_path),
        "pages": pages,
        "chunks": rag_doc["chunks"],
        "history": [],
        "artifacts": rag_doc["artifacts"],
        "model_name": model_name,
        "rag": {
            "document_id": chat_id,
            "index_path": rag_doc["artifacts"]["index"],
            "chunk_path": rag_doc["artifacts"]["chunks"],
            "model_name": model_name,
        },
    }


def _try_llm_answer(question: str, hits: list[dict], answer_model: str, history: list[dict] | None = None) -> str | None:
    """Try to answer via OpenRouter LLM. Returns None if unavailable.

    When history is provided, earlier turns are included as messages so the
    LLM can resolve pronouns and references like "that retriever".
    """
    import os as _os

    api_key = _os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None

    try:
        import requests as _requests

        context = "\n\n".join(
            f"### [Page {h['page']}]\n{h['text']}" for h in hits
        )
        system = (
            "You answer messages only from the supplied PDF text. "
            "Cite factual claims with [Page X]. "
            "If the answer is not in the PDF, say that the document does not provide enough information. "
            "Never invent a page number."
        )

        messages: list[dict] = [{"role": "system", "content": system}]

        # Include recent conversation history so follow-up questions have context
        if history:
            for turn in history[-4:]:
                messages.append({"role": "user", "content": turn.get("question", "")})
                messages.append({"role": "assistant", "content": turn.get("answer", "")})

        messages.append({"role": "user", "content": f"PDF text:\n{context}\n\nquestion: {question}"})

        model_id = _os.getenv("OPENROUTER_MODEL", answer_model)

        resp = _requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "temperature": 0.0,
                "messages": messages,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""
    except Exception:
        return None


def append_history(document: dict, question: str, result: dict) -> list[dict]:
    """Append a Q&A pair to the document's in-memory history and return the updated list."""
    entry = {
        "question": question,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "sources": result.get("sources", []),
    }
    if "history" not in document:
        document["history"] = []
    document["history"].append(entry)
    return document["history"]


# ── Evaluation Helpers ──────────────────────────────────────────────────────

def normalize_for_match(text: str) -> str:
    """Normalize text for simple string-based scoring.

    Lowercases, collapses whitespace, and strips punctuation that
    should not affect exact-match scoring.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Strip trailing punctuation marks
    text = text.rstrip(".,;:!?")
    return text


def contains_any_answer(text: str, answers: list[str]) -> bool:
    """Check whether any gold answer appears in the text after normalization."""
    norm_text = normalize_for_match(text)
    for ans in answers:
        norm_ans = normalize_for_match(ans)
        if norm_ans in norm_text:
            return True
    return False


def evaluate_questions(
    eval_set: list[dict],
    documents_by_name: dict[str, dict],
    top_k: int = 3,
    candidate_pool: int = 60,
):
    """Run evaluation over a set of questions and return a results DataFrame.

    Each row records: pdf_name, question, retrieved_pages, local_answer,
    retrieval_hit (gold answer in any retrieved chunk), and answer_hit
    (gold answer in the extracted answer string).
    """
    rows: list[dict] = []
    for item in eval_set:
        pdf_name = item["pdf_name"]
        question = item["question"]
        answers = item["answers"]
        document = documents_by_name.get(pdf_name)

        if document is None:
            rows.append({
                "pdf_name": pdf_name,
                "question": question,
                "retrieved_pages": [],
                "local_answer": "DOCUMENT NOT FOUND",
                "retrieval_hit": False,
                "answer_hit": False,
            })
            continue

        result = answer_document(document, question, top_k=top_k, candidate_pool=candidate_pool)
        hits = search_document(question, document, top_k=top_k, candidate_pool=candidate_pool)

        pages = sorted({h["page"] for h in hits})

        # retrieval_hit: any gold answer appears in retrieved chunk texts
        retrieval_hit = any(
            contains_any_answer(h["text"], answers) for h in hits
        )

        # answer_hit: gold answer appears in the extracted answer
        answer_hit = contains_any_answer(result["answer"], answers)

        rows.append({
            "pdf_name": pdf_name,
            "question": question,
            "retrieved_pages": pages,
            "local_answer": result["answer"],
            "retrieval_hit": retrieval_hit,
            "answer_hit": answer_hit,
        })

    import pandas as pd
    return pd.DataFrame(rows)


# ── Artifact Directory Helpers ───────────────────────────────────────────────

def ensure_artifact_dirs(artifact_root: str | Path | None = None) -> dict[str, Path]:
    """Return all artifact folder paths, creating them if needed.

    Returns a dict with keys: raw_pages, chunks, embeddings, reports, chroma, indexes.
    """
    if artifact_root is None:
        artifact_root = Path(__file__).resolve().parent.parent / "artifacts" / "rag"
    root = Path(artifact_root)
    dirs = {
        "raw_pages": root / "raw_pages",
        "chunks": root / "chunks",
        "embeddings": root / "embeddings",
        "reports": root / "reports",
        "chroma": root / "chroma",
        "indexes": root,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── Appendix B: Chroma Collection (Optional Branch) ──────────────────────────

def _require_chromadb():
    """Import chromadb or raise a clear ImportError."""
    try:
        import chromadb
        return chromadb
    except ImportError:
        raise ImportError(
            "chromadb is required for Chroma features. "
            "Install it with: pip install chromadb"
        )


def build_chroma_collection(
    document_id: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    persist_dir: str | Path,
) -> dict:
    """Build or reopen a persistent Chroma collection from chunks and embeddings.

    Stores page number and chunk_id as metadata so queries can return them directly.
    """
    chromadb = _require_chromadb()

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection_name = document_id

    # Delete existing collection with same name to rebuild cleanly
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"document_id": document_id},
    )

    # Prepare batch data
    ids = [str(c["chunk_id"]) for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"], "chunk_id": c["chunk_id"]} for c in chunks]
    emb_list = embeddings.tolist()

    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=emb_list)

    return {
        "collection_name": collection_name,
        "item_count": collection.count(),
    }


def query_chroma_collection(
    document_id: str,
    query_embedding: np.ndarray,
    persist_dir: str | Path,
    top_k: int,
) -> list[dict]:
    """Query a Chroma collection and return top-k matches.

    Each result carries: chunk_id, page, text, score.
    """
    chromadb = _require_chromadb()

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(document_id)

    q_emb = query_embedding.reshape(1, -1).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)

    hits: list[dict] = []
    if results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            doc_text = results["documents"][0][i] if results["documents"] else ""
            distance = results["distances"][0][i] if results["distances"] else 0.0
            hits.append({
                "chunk_id": meta.get("chunk_id", int(doc_id)),
                "page": meta.get("page", 0),
                "text": doc_text or "",
                "score": float(1.0 / (1.0 + distance)),
            })

    return hits


def search_document_with_chroma(
    question: str,
    document: dict,
    persist_dir: str | Path,
    top_k: int = 3,
    batch_size: int = 1,
) -> list[dict]:
    """Search via Chroma: embed question → query collection → return top-k hits."""
    q_embedding = embed_texts(
        [question],
        model_name=document.get("model_name", EMBEDDING_MODEL),
        batch_size=batch_size,
    )
    return query_chroma_collection(
        document_id=document["document_id"],
        query_embedding=q_embedding,
        persist_dir=persist_dir,
        top_k=top_k,
    )


def answer_document_with_chroma(
    document: dict,
    question: str,
    persist_dir: str | Path,
    top_k: int = 3,
    answer_model: str = "openrouter/free",
) -> dict:
    """Answer via Chroma retrieval. Returns the same {answer, citations, sources} shape as FAISS."""
    hits = search_document_with_chroma(question, document, persist_dir, top_k=top_k)

    answer = _try_llm_answer(question, hits, answer_model)
    if answer is None:
        answer = best_sentence_answer(question, hits)

    citations = extract_citations(answer, hits)
    sources = build_sources(hits)

    return {
        "answer": answer,
        "citations": citations,
        "sources": sources,
    }


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
