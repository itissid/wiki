"""ChromaDB + BM25 dual indexing for wiki chunks."""

from __future__ import annotations

import pickle
from pathlib import Path

import chromadb
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from src.chunker import WikiChunk, chunk_wiki

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
    chunks = chunk_wiki(wiki_text)
    if not chunks:
        raise ValueError(f"No chunks produced from wiki text for {repo}")

    repo_path = _repo_dir(repo, data_dir)
    repo_path.mkdir(parents=True, exist_ok=True)

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

    # Embed all chunk texts
    all_texts = [c.text for c in chunks]
    all_embeddings = _embed_texts(all_texts)

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
    tokenized_corpus = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(repo_path / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(repo_path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return len(chunks)


def load_index(
    repo: str, data_dir: str = "./data"
) -> tuple[chromadb.Collection, BM25Okapi, list[WikiChunk]]:
    """Load both indexes for a repo. Raises FileNotFoundError if not indexed."""
    repo_path = _repo_dir(repo, data_dir)

    if not is_indexed(repo, data_dir):
        raise FileNotFoundError(f"No index found for {repo} at {repo_path}")

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

    return collection, bm25, chunks
