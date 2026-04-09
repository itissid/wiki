"""MCP server exposing wiki ask_question tool."""

import asyncio
import os

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server

from src.ask import ask_question

DATA_DIR = os.environ.get("DATA_DIR", "./data")

server = Server("wiki-mcp")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="ask_question",
            description=(
                "Ask any question about a GitHub repository and get an AI-powered, "
                "context-grounded response. Auto-indexes the repo's wiki on first query."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repoName": {
                        "anyOf": [
                            {"type": "string"},
                            {"items": {"type": "string"}, "type": "array"},
                        ],
                        "description": "GitHub repository in owner/repo format",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to ask about the repository",
                    },
                },
                "required": ["repoName", "question"],
            },
        )
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name != "ask_question":
        raise ValueError(f"Unknown tool: {name}")

    repo_name = arguments["repoName"]
    question = arguments["question"]

    # Handle single repo or list (match DeepWiki API)
    if isinstance(repo_name, list):
        repo_name = repo_name[0]

    # Run sync ask_question in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, ask_question, repo_name, question, DATA_DIR
    )

    return [types.TextContent(type="text", text=result["answer"])]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
