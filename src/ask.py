"""Top-level ask function: orchestrates fetch -> index -> retrieve -> generate."""

from __future__ import annotations

import logging
import time

from src.fetcher import fetch_wiki
from src.generator import generate_answer
from src.indexer import build_index, is_indexed, load_index
from src.retriever import hybrid_search

logger = logging.getLogger("wiki-mcp")


def ask_question(
    repo: str,
    question: str,
    data_dir: str = "./data",
    n_results: int = 10,
    model: str = "sonnet",
) -> dict:
    """Ask a question about a repo. Auto-indexes if needed. Returns WikiAnswer-compatible dict."""

    logger.info("ask_question: repo=%s question=%r", repo, question[:80])

    # Step 1: Ensure repo is indexed
    if not is_indexed(repo, data_dir):
        logger.info("Repo %s not indexed yet — fetching and building index", repo)
        wiki_text = fetch_wiki(repo)
        build_index(repo, wiki_text, data_dir)
    else:
        logger.info("Repo %s already indexed — skipping fetch", repo)

    # Step 2: Load indexes
    collection, bm25, chunks = load_index(repo, data_dir)

    # Step 3: Hybrid retrieval
    retrieval_start = time.time()
    results = hybrid_search(
        question, collection, bm25, chunks, n_results=n_results
    )
    retrieval_time_ms = (time.time() - retrieval_start) * 1000

    # Step 4: Generate answer
    answer_text, generation_time_ms = generate_answer(
        question, results, model=model
    )

    return {
        "answer": answer_text,
        "repo": repo,
        "question": question,
        "chunks_used": len(results),
        "retrieval_time_ms": retrieval_time_ms,
        "generation_time_ms": generation_time_ms,
    }
