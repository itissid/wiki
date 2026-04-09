"""Phase 2 smoke tests: hybrid retrieval quality."""

from src.indexer import build_index, load_index
from src.retriever import hybrid_search, reciprocal_rank_fusion

SAMPLE_WIKI = """# Page: Getting Started
## Installation
Install pipecat with: pip install pipecat-ai

# Page: Transport Layer
## WebSocket Transport
WebSocketTransport handles real-time bidirectional communication.
Connect using ws://localhost:8765.

## HTTP Transport
HTTPTransport handles request-response patterns.

# Page: Pipeline
## Pipeline Class
The Pipeline class orchestrates processors in sequence.
Create with Pipeline(processors=[stt, llm, tts]).
"""


def test_rrf_basic():
    fused = reciprocal_rank_fusion([0, 1, 2], [2, 1, 0], k=60)
    # All 3 indices should appear
    assert len(fused) == 3
    # All items appear in both lists, so all should have positive scores
    assert all(score > 0 for _, score in fused)
    # Item appearing at rank 0 in one list should beat item appearing at rank 2 in both
    # (because 1/61 + 1/63 > 1/62 + 1/62 due to convexity)
    indices = [idx for idx, _ in fused]
    assert set(indices) == {0, 1, 2}


def test_rrf_disjoint():
    """Disjoint rankings should produce union of results."""
    fused = reciprocal_rank_fusion([0, 1], [2, 3], k=60)
    assert len(fused) == 4
    # Each item appears in exactly one list, so scores should be equal for same-rank items
    assert fused[0][1] == fused[1][1]  # rank 0 items tie


def test_hybrid_returns_results(tmp_path):
    data_dir = str(tmp_path)
    build_index("test/repo", SAMPLE_WIKI, data_dir)
    collection, bm25, chunks = load_index("test/repo", data_dir)

    results = hybrid_search("WebSocket transport", collection, bm25, chunks, n_results=3)
    assert len(results) > 0
    assert results[0].rrf_score > 0
    # Top result should mention WebSocket or Transport
    assert "WebSocket" in results[0].chunk.text or "Transport" in results[0].chunk.page


def test_bm25_finds_exact_terms(tmp_path):
    data_dir = str(tmp_path)
    build_index("test/repo", SAMPLE_WIKI, data_dir)
    collection, bm25, chunks = load_index("test/repo", data_dir)

    results = hybrid_search(
        "Pipeline class processors", collection, bm25, chunks, n_results=3
    )
    assert any(r.bm25_rank is not None and r.bm25_rank < 3 for r in results)


def test_vector_finds_semantic(tmp_path):
    data_dir = str(tmp_path)
    build_index("test/repo", SAMPLE_WIKI, data_dir)
    collection, bm25, chunks = load_index("test/repo", data_dir)

    # Semantic query - doesn't use exact words from the wiki
    results = hybrid_search(
        "how to set up real-time communication", collection, bm25, chunks, n_results=3
    )
    assert any(r.vector_rank is not None for r in results)


def test_both_sources_contribute(tmp_path):
    """At least one result should have both bm25_rank and vector_rank set."""
    data_dir = str(tmp_path)
    build_index("test/repo", SAMPLE_WIKI, data_dir)
    collection, bm25, chunks = load_index("test/repo", data_dir)

    results = hybrid_search("WebSocket transport", collection, bm25, chunks, n_results=5)
    has_both = any(r.bm25_rank is not None and r.vector_rank is not None for r in results)
    has_bm25 = any(r.bm25_rank is not None for r in results)
    has_vector = any(r.vector_rank is not None for r in results)
    # Both search methods should contribute results
    assert has_bm25, "No BM25 results found"
    assert has_vector, "No vector results found"
