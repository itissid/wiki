"""Hybrid BM25 + Vector retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.chunker import WikiChunk
from src.indexer import _embed_texts

logger = logging.getLogger("wiki-mcp")


@dataclass
class RetrievalResult:
    """A single retrieval result with provenance from both search methods."""

    chunk: WikiChunk
    bm25_rank: int | None
    vector_rank: int | None
    rrf_score: float


def reciprocal_rank_fusion(
    bm25_ranking: list[int],
    vector_ranking: list[int],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge two rankings using RRF. Returns [(chunk_index, rrf_score), ...] sorted descending."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranking):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(vector_ranking):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(
    query: str,
    collection,  # chromadb.Collection
    bm25: BM25Okapi,
    chunks: list[WikiChunk],
    n_results: int = 10,
    bm25_candidates: int = 25,
    vector_candidates: int = 25,
) -> list[RetrievalResult]:
    """Run hybrid search and return top-n results via RRF."""
    logger.info("Hybrid search: query=%r, n_results=%d", query[:80], n_results)
    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:bm25_candidates]

    # Vector search
    query_embedding = _embed_texts([query])
    vector_results = collection.query(
        query_embeddings=query_embedding, n_results=min(vector_candidates, collection.count())
    )
    # Map ChromaDB IDs back to chunk indices
    vector_top = [int(doc_id.split("_")[1]) for doc_id in vector_results["ids"][0]]

    # Merge via RRF
    fused = reciprocal_rank_fusion(bm25_top, vector_top)

    logger.info("RRF fusion produced %d candidates, returning top %d", len(fused), n_results)
    # Build RetrievalResult objects
    results = []
    for chunk_idx, rrf_score in fused[:n_results]:
        bm25_r = bm25_top.index(chunk_idx) if chunk_idx in bm25_top else None
        vector_r = vector_top.index(chunk_idx) if chunk_idx in vector_top else None
        results.append(
            RetrievalResult(
                chunk=chunks[chunk_idx],
                bm25_rank=bm25_r,
                vector_rank=vector_r,
                rrf_score=rrf_score,
            )
        )
    return results
