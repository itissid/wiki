"""Fetch wiki content from DeepWiki via direct MCP client call."""

from __future__ import annotations

import asyncio
import logging

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEEPWIKI_URL = "https://mcp.deepwiki.com/mcp"

logger = logging.getLogger("wiki-mcp")


def fetch_wiki(repo: str) -> str:
    """Fetch full wiki markdown for a repo by calling DeepWiki's read_wiki_contents directly.

    Uses the MCP Python SDK to make a direct HTTP call to DeepWiki's MCP endpoint.
    No Claude CLI middleman — just a straightforward tool call.
    This is a one-time cost per repo (wiki is cached locally after indexing).
    """
    return asyncio.run(_fetch_wiki_async(repo))


async def _fetch_wiki_async(repo: str) -> str:
    logger.info("Fetching wiki for %s from DeepWiki (direct MCP call)...", repo)

    async with streamablehttp_client(DEEPWIKI_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "read_wiki_contents", {"repoName": repo}
            )

    wiki_text = result.content[0].text
    logger.info("Fetched wiki for %s: %d chars", repo, len(wiki_text))

    if not wiki_text or len(wiki_text) < 500:
        raise ValueError(
            f"Wiki content too short for {repo} ({len(wiki_text)} chars). "
            "DeepWiki may not have indexed this repo yet."
        )

    return wiki_text
