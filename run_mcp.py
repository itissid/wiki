"""Run the MCP server locally for testing (stdio mode)."""

import asyncio
import logging
import os
import sys

# Configure logging to stderr (stdout is the MCP protocol channel)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

os.environ.setdefault("DATA_DIR", "./data")

from src.mcp_server import main

asyncio.run(main())
