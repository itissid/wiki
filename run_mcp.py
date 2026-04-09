"""Run the MCP server locally for testing (stdio mode)."""

import asyncio
import os

os.environ.setdefault("DATA_DIR", "./data")

from src.mcp_server import main

asyncio.run(main())
