"""Run the MCP server over HTTP (for Docker deployment).

Uses the MCP SDK's built-in StreamableHTTP transport with uvicorn.
Logs go to stderr which Docker captures as container logs.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.routing import Mount

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

# Configure logging before importing app modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

os.environ.setdefault("DATA_DIR", "./data")

from src.mcp_server import server  # noqa: E402

session_manager = StreamableHTTPSessionManager(app=server, stateless=True)


@asynccontextmanager
async def lifespan(app):
    async with session_manager.run():
        yield


app = Starlette(
    routes=[
        Mount("/mcp", app=session_manager.handle_request),
    ],
    lifespan=lifespan,
)

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
