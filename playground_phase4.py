"""Phase 4 playground: Test all 3 MCP tools via the MCP client SDK."""

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
            init = await session.initialize()
            print(f"Server: {init.serverInfo.name}")
            print(f"Protocol: {init.protocolVersion}")
            print()

            # List tools
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print(f"Tool: {tool.name}")
                print(f"  Description: {tool.description}")
            print()

            repo = "pipecat-ai/pipecat"

            # Test read_wiki_structure
            print(f"--- read_wiki_structure({repo}) ---")
            print("(This may take 30-60s on first run while wiki is fetched and indexed)")
            result = await session.call_tool("read_wiki_structure", {"repoName": repo})
            print(result.content[0].text[:500])
            print()

            # Test read_wiki_contents
            print(f"--- read_wiki_contents({repo}) ---")
            result = await session.call_tool("read_wiki_contents", {"repoName": repo})
            print(f"Content length: {len(result.content[0].text)} chars")
            print(f"Preview: {result.content[0].text[:200]}...")
            print()

            # Test ask_question
            print(f"--- ask_question({repo}, 'How do I install pipecat?') ---")
            result = await session.call_tool(
                "ask_question",
                {"repoName": repo, "question": "How do I install pipecat?"},
            )
            print(f"Answer:\n{result.content[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
