"""Answer generation via Claude Code CLI binary."""

from __future__ import annotations

import json
import subprocess
import time

from src.retriever import RetrievalResult

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a GitHub repository's documentation. "
    "Answer based ONLY on the provided context. If the context does not contain enough information "
    "to answer the question, say so clearly. Be concise and specific. "
    "When referencing specific features, include the relevant page/section name."
)


def generate_answer(
    question: str,
    context_chunks: list[RetrievalResult],
    model: str = "sonnet",
) -> tuple[str, float]:
    """Generate answer using claude CLI. Returns (answer_text, generation_time_ms)."""

    # Build context string from retrieved chunks
    context_parts = []
    for i, result in enumerate(context_chunks):
        context_parts.append(
            f"[Source {i + 1}: {result.chunk.page} > {result.chunk.heading}]\n"
            f"{result.chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"Question: {question}"

    start = time.time()
    result = subprocess.run(
        [
            "claude",
            "-p",
            prompt,
            "--model",
            model,
            "--output-format",
            "json",
            "--no-session-persistence",
            "--max-turns",
            "1",
            "--system-prompt",
            SYSTEM_PROMPT,
        ],
        input=context,
        capture_output=True,
        text=True,
        timeout=120,
    )
    generation_time_ms = (time.time() - start) * 1000

    if result.returncode != 0:
        raise RuntimeError(
            f"claude exited {result.returncode}: {result.stderr[:500]}"
        )

    data = json.loads(result.stdout)
    return data["result"], generation_time_ms
