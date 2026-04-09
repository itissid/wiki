"""Answer generation via Anthropic SDK or Claude Code CLI."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time

from src.retriever import RetrievalResult

logger = logging.getLogger("wiki-mcp")

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a GitHub repository's documentation. "
    "Answer based ONLY on the provided context. If the context does not contain enough information "
    "to answer the question, say so clearly. Be concise and specific. "
    "When referencing specific features, include the relevant page/section name."
)

# Map short model names to Anthropic API model IDs
_MODEL_MAP = {
    "sonnet": "claude-sonnet-4-20250514",
    "opus": "claude-opus-4-20250514",
    "haiku": "claude-haiku-4-5-20251001",
}


def _build_context(question: str, context_chunks: list[RetrievalResult]) -> tuple[str, str]:
    """Build context string and prompt from chunks. Returns (context, prompt)."""
    context_parts = []
    for i, result in enumerate(context_chunks):
        context_parts.append(
            f"[Source {i + 1}: {result.chunk.page} > {result.chunk.heading}]\n"
            f"{result.chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)
    prompt = f"Question: {question}"
    return context, prompt


def _generate_sdk(context: str, prompt: str, model: str) -> str:
    """Generate answer using the Anthropic Python SDK."""
    import anthropic

    client = anthropic.Anthropic()
    model_id = _MODEL_MAP.get(model, model)
    response = client.messages.create(
        model=model_id,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"{context}\n\n---\n\n{prompt}"},
        ],
    )
    return response.content[0].text


def _generate_cli(context: str, prompt: str, model: str) -> str:
    """Generate answer using the Claude Code CLI binary."""
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
    if result.returncode != 0:
        raise RuntimeError(
            f"claude exited {result.returncode}\n"
            f"stderr: {result.stderr[:500]}\n"
            f"stdout: {result.stdout[:500]}"
        )
    data = json.loads(result.stdout)
    return data["result"]


def generate_answer(
    question: str,
    context_chunks: list[RetrievalResult],
    model: str = "sonnet",
) -> tuple[str, float]:
    """Generate answer. Uses Anthropic SDK if ANTHROPIC_API_KEY is set, else Claude CLI."""
    context, prompt = _build_context(question, context_chunks)
    logger.info("Generating answer with model=%s using %d context chunks (%d chars)", model, len(context_chunks), len(context))

    start = time.time()
    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Using Anthropic SDK (ANTHROPIC_API_KEY set)")
        answer = _generate_sdk(context, prompt, model)
    elif shutil.which("claude"):
        logger.info("Using Claude CLI")
        answer = _generate_cli(context, prompt, model)
    else:
        raise RuntimeError(
            "No generation backend available. Set ANTHROPIC_API_KEY or install Claude CLI."
        )
    generation_time_ms = (time.time() - start) * 1000

    logger.info("Generation complete in %.0fms", generation_time_ms)
    return answer, generation_time_ms
