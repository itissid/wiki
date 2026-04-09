"""Phase 4 tests: MCP server protocol compliance.

Tests 1-2 use the real MCP server over stdio (subprocess).
Tests 3-7 use in-process testing with mocks for tool logic.
"""

import asyncio
from unittest.mock import patch

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


SERVER_PARAMS = StdioServerParameters(
    command="uv",
    args=["run", "python", "run_mcp.py"],
    cwd="/home/dev/workspace/wiki",
)


@pytest.fixture
def run():
    """Helper to run async test functions."""
    def _run(coro):
        return asyncio.run(coro)
    return _run


def test_mcp_initialize(run):
    """Server responds to initialize with correct server info."""
    async def _test():
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                result = await session.initialize()
                assert result.serverInfo.name == "wiki-mcp"
    run(_test())


def test_mcp_list_tools(run):
    """tools/list returns all 3 tools with correct schemas."""
    async def _test():
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tools = tools_result.tools
                assert len(tools) == 3
                names = {t.name for t in tools}
                assert names == {"read_wiki_structure", "read_wiki_contents", "ask_question"}
                # All tools have repoName
                for t in tools:
                    assert "repoName" in t.inputSchema["properties"]
                # ask_question additionally has question
                ask_tool = next(t for t in tools if t.name == "ask_question")
                assert "question" in ask_tool.inputSchema["properties"]
    run(_test())


def test_mcp_call_ask_question_mock(run):
    """call_tool ask_question returns text content (in-process with mock)."""
    async def _test():
        mock_result = {
            "answer": "Pipecat is installed via pip install pipecat-ai",
            "repo": "pipecat-ai/pipecat",
            "question": "How do I install pipecat?",
            "chunks_used": 5,
            "retrieval_time_ms": 42.0,
            "generation_time_ms": 1500.0,
        }
        with patch("src.mcp_server.ask_question", return_value=mock_result):
            from src.mcp_server import call_tool
            result = await call_tool(
                "ask_question",
                {"repoName": "pipecat-ai/pipecat", "question": "How do I install pipecat?"},
            )
            assert len(result) == 1
            assert result[0].type == "text"
            assert "pipecat-ai" in result[0].text
    run(_test())


def test_mcp_call_read_wiki_structure_mock(run):
    """call_tool read_wiki_structure returns structure text."""
    async def _test():
        mock_structure = {
            "Getting Started": ["Installation", "Quick Start"],
            "Transport Layer": ["WebSocket Transport", "HTTP Transport"],
        }
        with patch("src.mcp_server._ensure_indexed"), \
             patch("src.mcp_server.get_wiki_structure", return_value=mock_structure), \
             patch("src.mcp_server.list_indexed_repos", return_value=["test/repo"]):
            from src.mcp_server import call_tool
            result = await call_tool("read_wiki_structure", {"repoName": "test/repo"})
            assert len(result) == 1
            assert result[0].type == "text"
            assert "Getting Started" in result[0].text
            assert "WebSocket Transport" in result[0].text
    run(_test())


def test_mcp_call_read_wiki_contents_mock(run):
    """call_tool read_wiki_contents returns reconstructed wiki text."""
    async def _test():
        from src.chunker import WikiChunk
        mock_chunks = [
            WikiChunk(text="Install with pip", page="Getting Started", heading="Install", chunk_index=0),
            WikiChunk(text="Create a pipeline", page="Getting Started", heading="Quick Start", chunk_index=1),
            WikiChunk(text="WebSocket docs", page="Transport", heading="WebSocket", chunk_index=2),
        ]
        with patch("src.mcp_server._ensure_indexed"), \
             patch("src.mcp_server.load_index", return_value=(None, None, mock_chunks)):
            from src.mcp_server import call_tool
            result = await call_tool("read_wiki_contents", {"repoName": "test/repo"})
            assert len(result) == 1
            assert result[0].type == "text"
            text = result[0].text
            assert "# Page: Getting Started" in text
            assert "# Page: Transport" in text
            assert "Install with pip" in text
    run(_test())


def test_mcp_call_tool_repo_list(run):
    """call_tool handles repoName as a list (uses first element)."""
    async def _test():
        mock_result = {
            "answer": "Test answer",
            "repo": "owner/repo",
            "question": "test",
            "chunks_used": 1,
            "retrieval_time_ms": 1.0,
            "generation_time_ms": 1.0,
        }
        with patch("src.mcp_server.ask_question", return_value=mock_result) as mock_fn:
            from src.mcp_server import call_tool
            await call_tool(
                "ask_question",
                {"repoName": ["owner/repo", "owner/repo2"], "question": "test"},
            )
            mock_fn.assert_called_once()
            assert mock_fn.call_args[0][0] == "owner/repo"
    run(_test())


def test_mcp_call_tool_unknown_tool(run):
    """call_tool raises ValueError for unknown tool names."""
    async def _test():
        from src.mcp_server import call_tool
        with pytest.raises(ValueError, match="Unknown tool"):
            await call_tool("nonexistent_tool", {})
    run(_test())
