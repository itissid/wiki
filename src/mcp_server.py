"""MCP server exposing wiki tools: read_wiki_structure, read_wiki_contents, ask_question."""

import asyncio
import json
import logging
import os

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server

from src.ask import ask_question
from src.fetcher import fetch_wiki
from src.indexer import (
    build_index,
    get_wiki_structure,
    is_indexed,
    list_indexed_repos,
    load_index,
)

logger = logging.getLogger("wiki-mcp")

DATA_DIR = os.environ.get("DATA_DIR", "./data")

server = Server("wiki-mcp")

REPO_NAME_SCHEMA = {
    "anyOf": [
        {"type": "string"},
        {"items": {"type": "string"}, "type": "array"},
    ],
    "description": "GitHub repository in owner/repo format",
}


def _normalize_repo(repo_name) -> str:
    """Normalize repoName: if list, use first element."""
    if isinstance(repo_name, list):
        logger.info("repoName is a list, using first: %s", repo_name[0])
        return repo_name[0]
    return repo_name


def _ensure_indexed(repo: str) -> None:
    """Fetch and index a repo if not already indexed."""
    if not is_indexed(repo, DATA_DIR):
        logger.info("Repo %s not indexed — fetching from DeepWiki and indexing", repo)
        wiki_text = fetch_wiki(repo)
        build_index(repo, wiki_text, DATA_DIR)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="read_wiki_structure",
            description=(
                "Get the table of contents for a GitHub repository's wiki. "
                "Returns page names and section headings. "
                "Auto-fetches and indexes the repo on first call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repoName": REPO_NAME_SCHEMA,
                },
                "required": ["repoName"],
            },
        ),
        types.Tool(
            name="read_wiki_contents",
            description=(
                "Get the full wiki documentation for a GitHub repository. "
                "Returns the complete markdown content of all wiki pages. "
                "Auto-fetches and indexes the repo on first call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repoName": REPO_NAME_SCHEMA,
                },
                "required": ["repoName"],
            },
        ),
        types.Tool(
            name="ask_question",
            description=(
                "Ask any question about a GitHub repository and get an AI-powered, "
                "context-grounded response using hybrid RAG (BM25 + vector search). "
                "Auto-indexes the repo's wiki on first query."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repoName": REPO_NAME_SCHEMA,
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the repository",
                    },
                },
                "required": ["repoName", "question"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:

    if name == "read_wiki_structure":
        repo = _normalize_repo(arguments["repoName"])
        logger.info("MCP call_tool: read_wiki_structure repo=%s", repo)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _ensure_indexed, repo)
        structure = await loop.run_in_executor(
            None, get_wiki_structure, repo, DATA_DIR
        )

        # Format as readable text
        lines = [f"Wiki structure for {repo}:", f"({sum(len(v) for v in structure.values())} sections across {len(structure)} pages)", ""]
        for page, headings in structure.items():
            lines.append(f"## {page}")
            for heading in headings:
                if heading != page:
                    lines.append(f"  - {heading}")
            lines.append("")

        indexed_repos = await loop.run_in_executor(None, list_indexed_repos, DATA_DIR)
        if len(indexed_repos) > 1:
            lines.append(f"Other indexed repos: {', '.join(r for r in indexed_repos if r != repo)}")

        text = "\n".join(lines)
        logger.info("read_wiki_structure complete: %d pages, %d sections", len(structure), sum(len(v) for v in structure.values()))
        return [types.TextContent(type="text", text=text)]

    elif name == "read_wiki_contents":
        repo = _normalize_repo(arguments["repoName"])
        logger.info("MCP call_tool: read_wiki_contents repo=%s", repo)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _ensure_indexed, repo)

        # Load chunks and reconstruct full content
        _, _, chunks = await loop.run_in_executor(None, load_index, repo, DATA_DIR)

        # Group chunks by page, preserving order
        pages: dict[str, list[str]] = {}
        for chunk in chunks:
            if chunk.page not in pages:
                pages[chunk.page] = []
            pages[chunk.page].append(chunk.text)

        # Reconstruct markdown
        parts = []
        for page_title, texts in pages.items():
            parts.append(f"# Page: {page_title}\n")
            parts.append("\n\n".join(texts))
            parts.append("")

        text = "\n".join(parts)
        logger.info("read_wiki_contents complete: %d pages, %d chars", len(pages), len(text))
        return [types.TextContent(type="text", text=text)]

    elif name == "ask_question":
        repo_name = _normalize_repo(arguments["repoName"])
        question = arguments["question"]

        logger.info("MCP call_tool: ask_question repo=%s question=%r", repo_name, question[:80])

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, ask_question, repo_name, question, DATA_DIR
        )

        logger.info(
            "MCP ask_question complete: %d chunks used, retrieval=%.0fms, generation=%.0fms",
            result["chunks_used"], result["retrieval_time_ms"], result["generation_time_ms"],
        )
        return [types.TextContent(type="text", text=result["answer"])]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
