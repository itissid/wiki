"""Markdown-aware wiki chunking for DeepWiki output."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class WikiChunk:
    """A chunk of wiki content with metadata."""

    text: str
    page: str
    heading: str
    chunk_index: int


def chunk_wiki(
    wiki_text: str,
    max_chars: int = 1500,
    overlap: int = 150,
) -> list[WikiChunk]:
    """Split DeepWiki wiki output into chunks with metadata.

    Stage 1: Split on ``# Page:`` delimiters, then by ``##``/``###`` headings.
    Stage 2: Split oversized sections by paragraph with overlap.
    Code blocks are preserved as atomic units.
    """
    pages = re.split(r"(?=^# Page: )", wiki_text, flags=re.MULTILINE)
    chunks: list[WikiChunk] = []
    idx = 0

    for page in pages:
        page = page.strip()
        if not page:
            continue

        page_match = re.match(r"# Page: (.+)", page)
        page_title = page_match.group(1).strip() if page_match else "intro"

        # Split by h2/h3 within each page
        sections = re.split(r"\n(?=#{2,3} )", page)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            heading_match = re.match(r"(#{2,3}) (.+)", section)
            heading = heading_match.group(2).strip() if heading_match else page_title

            if len(section) <= max_chars:
                chunks.append(
                    WikiChunk(
                        text=section,
                        page=page_title,
                        heading=heading,
                        chunk_index=idx,
                    )
                )
                idx += 1
            else:
                # Further split by paragraph, preserving code blocks
                paragraphs = _split_preserving_code_blocks(section)
                buf = ""
                for p in paragraphs:
                    if len(buf) + len(p) > max_chars and buf:
                        chunks.append(
                            WikiChunk(
                                text=buf.strip(),
                                page=page_title,
                                heading=heading,
                                chunk_index=idx,
                            )
                        )
                        idx += 1
                        # Keep overlap from end of previous chunk
                        buf = buf[-overlap:] + "\n\n" + p
                    else:
                        buf += "\n\n" + p if buf else p
                if buf.strip():
                    chunks.append(
                        WikiChunk(
                            text=buf.strip(),
                            page=page_title,
                            heading=heading,
                            chunk_index=idx,
                        )
                    )
                    idx += 1

    return chunks


def _split_preserving_code_blocks(text: str) -> list[str]:
    """Split text by double-newline but keep code blocks (``` ... ```) as atomic units."""
    parts: list[str] = []
    current = ""
    in_code_block = False

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            current += line + "\n"
            continue

        if in_code_block:
            current += line + "\n"
            continue

        # Outside code block: split on blank lines
        if line.strip() == "":
            if current.strip():
                parts.append(current.strip())
                current = ""
        else:
            current += line + "\n"

    if current.strip():
        parts.append(current.strip())

    return parts
