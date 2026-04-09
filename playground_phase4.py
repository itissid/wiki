"""Phase 4 playground: Test MCP server via the MCP client SDK over stdio."""

import asyncio

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

SERVER_PARAMS = StdioServerParameters(
    command="uv",
    args=["run", "python", "run_mcp.py"],
    cwd="/home/dev/workspace/wiki",
)


async def main():
    print("Connecting to wiki-mcp server...")
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            init = await session.initialize()
            print(f"Server: {init.serverInfo.name}")
            print(f"Protocol: {init.protocolVersion}")
            print()

            # List tools
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print(f"Tool: {tool.name}")
                print(f"  Description: {tool.description}")
                print(f"  Schema: {tool.inputSchema}")
            print()

            # Ask a question (this will hit the real pipeline)
            print("Calling ask_question for pipecat-ai/pipecat...")
            print("(This may take 30-60s on first run while wiki is fetched and indexed)")
            result = await session.call_tool(
                "ask_question",
                {
                    "repoName": "pipecat-ai/pipecat",
                    "question": "How do I install pipecat?",
                },
            )
            for content in result.content:
                print(f"\nAnswer:\n{content.text}")


if __name__ == "__main__":
    asyncio.run(main())
