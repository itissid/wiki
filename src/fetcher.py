"""Fetch wiki content from DeepWiki via the claude CLI."""

from __future__ import annotations

import json
import subprocess


def fetch_wiki(repo: str) -> str:
    """Fetch full wiki markdown for a repo using DeepWiki's read_wiki_contents via claude CLI.

    This leverages the existing DeepWiki MCP connection configured in ~/.codex/config.toml.
    It's a one-time cost per repo (wiki is cached locally after indexing).
    """
    prompt = (
        f"Use the mcp__deepwiki__read_wiki_contents tool to get the full wiki for {repo}. "
        "Return ONLY the raw wiki content text, nothing else. No commentary, no markdown "
        "formatting around it, just the wiki text as-is."
    )

    result = subprocess.run(
        [
            "claude",
            "-p",
            prompt,
            "--model",
            "haiku",
            "--output-format",
            "json",
            "--max-turns",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 min timeout for large wikis
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to fetch wiki for {repo}: claude exited {result.returncode}\n"
            f"stderr: {result.stderr[:500]}"
        )

    data = json.loads(result.stdout)
    wiki_text = data.get("result", "")

    if not wiki_text or len(wiki_text) < 100:
        raise ValueError(
            f"Wiki content too short for {repo} ({len(wiki_text)} chars). "
            "DeepWiki may not have indexed this repo yet."
        )

    return wiki_text
