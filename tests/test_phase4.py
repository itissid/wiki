"""Phase 4 tests: MCP server protocol compliance.

Tests 1-2 use the real MCP server over stdio (subprocess).
Test 3 uses in-process testing with mocked ask_question to verify call_tool.
"""

import asyncio
from unittest.mock import patch, AsyncMock

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
    """tools/list returns ask_question with correct schema."""
    async def _test():
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tools = tools_result.tools
                assert len(tools) == 1
                tool = tools[0]
                assert tool.name == "ask_question"
                schema = tool.inputSchema
                assert "repoName" in schema["properties"]
                assert "question" in schema["properties"]
                assert set(schema["required"]) == {"repoName", "question"}
    run(_test())


def test_mcp_call_tool_with_mock(run):
    """call_tool handler returns text content (in-process with mocked ask_question)."""
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
            # Verify first repo was used
            mock_fn.assert_called_once()
            args = mock_fn.call_args[0]
            assert args[0] == "owner/repo"
    run(_test())


def test_mcp_call_tool_unknown_tool(run):
    """call_tool raises ValueError for unknown tool names."""
    async def _test():
        from src.mcp_server import call_tool
        with pytest.raises(ValueError, match="Unknown tool"):
            await call_tool("nonexistent_tool", {})
    run(_test())
