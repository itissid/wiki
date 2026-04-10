"""ChromaDB + BM25 dual indexing for wiki chunks."""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

import chromadb
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from src.chunker import WikiChunk, chunk_wiki

logger = logging.getLogger("wiki-mcp")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "wiki"

# Module-level cache so we only load the model once per process
_embed_model: TextEmbedding | None = None


def _get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embed_model


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using FastEmbed."""
    model = _get_embed_model()
    return [e.tolist() for e in model.embed(texts)]


def _repo_slug(repo: str) -> str:
    """Convert 'owner/repo' to 'owner_repo' for filesystem paths."""
    return repo.replace("/", "_")


def _repo_dir(repo: str, data_dir: str) -> Path:
    return Path(data_dir) / _repo_slug(repo)


def is_indexed(repo: str, data_dir: str = "./data") -> bool:
    """Check if a repo has already been indexed."""
    d = _repo_dir(repo, data_dir)
    return (d / "chromadb").exists() and (d / "bm25.pkl").exists()


def build_index(repo: str, wiki_text: str, data_dir: str = "./data") -> int:
    """Chunk wiki text and build both ChromaDB and BM25 indexes. Returns chunk count."""
    logger.info("Indexing %s (%d chars of wiki text)...", repo, len(wiki_text))
    chunks = chunk_wiki(wiki_text)
    if not chunks:
        raise ValueError(f"No chunks produced from wiki text for {repo}")

    repo_path = _repo_dir(repo, data_dir)
    repo_path.mkdir(parents=True, exist_ok=True)

    # Embed all chunk texts first — this can take minutes on CPU,
    # and ChromaDB collections go stale if there's a long gap between
    # create_collection() and add().
    logger.info("Embedding %d chunks with %s...", len(chunks), EMBEDDING_MODEL)
    all_texts = [c.text for c in chunks]
    all_embeddings = _embed_texts(all_texts)
    logger.info("Embeddings complete, building ChromaDB index...")

    # --- ChromaDB vector index ---
    chroma_path = str(repo_path / "chromadb")
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=chromadb.Settings(anonymized_telemetry=False, is_persistent=True),
    )
    # Delete collection if it already exists (rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection doesn't exist yet

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": EMBEDDING_MODEL},
    )

    # Add chunks in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_embeddings = all_embeddings[i : i + batch_size]
        collection.add(
            documents=[c.text for c in batch],
            embeddings=batch_embeddings,
            metadatas=[
                {"page": c.page, "heading": c.heading, "chunk_index": c.chunk_index}
                for c in batch
            ],
            ids=[f"chunk_{c.chunk_index}" for c in batch],
        )

    # --- BM25 keyword index ---
    logger.info("Building BM25 index...")
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(repo_path / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(repo_path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    logger.info("Index complete for %s: %d chunks stored at %s", repo, len(chunks), repo_path)

    # Export per-page markdown for external viewers (Obsidian, etc.)
    export_markdown(repo, data_dir)

    return len(chunks)


def _sanitize_filename(title: str) -> str:
    """Convert a wiki page title to a safe filename.

    Replaces characters invalid on macOS/Windows/Linux with underscores.
    Strips leading/trailing whitespace and dots.
    """
    safe = re.sub(r'[/\\:*?"<>|]', '_', title)
    safe = safe.strip('. ')
    safe = re.sub(r'_+', '_', safe)
    return safe or "untitled"


def export_markdown(repo: str, data_dir: str = "./data") -> int:
    """Export per-page .md files from an existing index's chunks.pkl.

    Writes to data/{repo}/pages/{PageTitle}.md.
    Returns the number of pages exported.
    """
    repo_path = _repo_dir(repo, data_dir)
    chunks_file = repo_path / "chunks.pkl"
    if not chunks_file.exists():
        raise FileNotFoundError(f"No chunks.pkl found for {repo} at {repo_path}")

    with open(chunks_file, "rb") as f:
        chunks: list[WikiChunk] = pickle.load(f)

    # Group chunks by page, preserving order
    pages: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.page not in pages:
            pages[chunk.page] = []
        pages[chunk.page].append(chunk.text)

    # Write per-page markdown files
    pages_dir = repo_path / "pages"
    # Clean out old pages on re-export
    if pages_dir.exists():
        for old_file in pages_dir.iterdir():
            if old_file.suffix == ".md":
                old_file.unlink()
    pages_dir.mkdir(parents=True, exist_ok=True)

    for page_title, texts in pages.items():
        filename = _sanitize_filename(page_title) + ".md"
        content = f"# {page_title}\n\n" + "\n\n".join(texts)
        (pages_dir / filename).write_text(content, encoding="utf-8")

    logger.info("Exported %d pages as markdown to %s", len(pages), pages_dir)
    return len(pages)


def list_indexed_repos(data_dir: str = "./data") -> list[str]:
    """Return list of repos that have been indexed (as owner/repo strings)."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    repos = []
    for d in sorted(data_path.iterdir()):
        if d.is_dir() and (d / "chromadb").exists() and (d / "bm25.pkl").exists():
            # Convert owner_repo back to owner/repo
            parts = d.name.split("_", 1)
            if len(parts) == 2:
                repos.append(f"{parts[0]}/{parts[1]}")
    return repos


def get_wiki_structure(repo: str, data_dir: str = "./data") -> dict[str, list[str]]:
    """Return page -> [headings] structure for an indexed repo."""
    if not is_indexed(repo, data_dir):
        return {}
    repo_path = _repo_dir(repo, data_dir)
    with open(repo_path / "chunks.pkl", "rb") as f:
        chunks: list[WikiChunk] = pickle.load(f)

    structure: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.page not in structure:
            structure[chunk.page] = []
        if chunk.heading not in structure[chunk.page]:
            structure[chunk.page].append(chunk.heading)
    return structure


def load_index(
    repo: str, data_dir: str = "./data"
) -> tuple[chromadb.Collection, BM25Okapi, list[WikiChunk]]:
    """Load both indexes for a repo. Raises FileNotFoundError if not indexed."""
    repo_path = _repo_dir(repo, data_dir)

    if not is_indexed(repo, data_dir):
        raise FileNotFoundError(f"No index found for {repo} at {repo_path}")

    logger.info("Loading existing index for %s from %s", repo, repo_path)

    # ChromaDB
    chroma_path = str(repo_path / "chromadb")
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=chromadb.Settings(anonymized_telemetry=False, is_persistent=True),
    )
    collection = client.get_collection(name=COLLECTION_NAME)

    # BM25
    with open(repo_path / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(repo_path / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    logger.info("Loaded index for %s: %d chunks", repo, len(chunks))
    return collection, bm25, chunks
