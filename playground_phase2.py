"""Phase 2 playground: Test hybrid retrieval on indexed wiki."""

from src.indexer import load_index, is_indexed, build_index
from src.retriever import hybrid_search

REPO = "pipecat-ai/pipecat"
DATA_DIR = "./data"

QUERIES = [
    "How does pipecat handle WebSocket transport?",
    "What is a pipeline in pipecat?",
    "How do I install pipecat?",
    "What LLM services does pipecat support?",
    "pipecat Pipeline class",  # keyword-heavy query (BM25 should shine)
]


def main():
    if not is_indexed(REPO, DATA_DIR):
        print(f"Index not found for {REPO}. Run playground_phase1.py first.")
        return

    collection, bm25, chunks = load_index(REPO, DATA_DIR)

    for query in QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        results = hybrid_search(query, collection, bm25, chunks, n_results=5)
        for i, r in enumerate(results):
            bm25_tag = f"BM25#{r.bm25_rank}" if r.bm25_rank is not None else "BM25:miss"
            vec_tag = f"Vec#{r.vector_rank}" if r.vector_rank is not None else "Vec:miss"
            print(f"  [{i+1}] {bm25_tag} | {vec_tag} | RRF={r.rrf_score:.4f}")
            print(f"      Page: {r.chunk.page} > {r.chunk.heading}")
            print(f"      {r.chunk.text[:120]}...")


if __name__ == "__main__":
    main()
