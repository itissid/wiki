FROM python:3.12-slim

WORKDIR /app

# Install system deps for chromadb (needs build tools for hnswlib) and Node.js for Claude CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files and install deps
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY baml_src/ ./baml_src/
RUN uv run baml-cli generate

COPY src/ ./src/
COPY run_mcp.py run_mcp_http.py ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8080

ENV DATA_DIR=/data
ENV PORT=8080

CMD ["uv", "run", "python", "run_mcp_http.py"]
