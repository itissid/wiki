# Wiki MCP Server

Local hybrid RAG server that replaces DeepWiki's `ask_question` with a
BM25 + vector search pipeline. Exposes 3 MCP tools over stdio and uses
Claude CLI for answer generation.

## Architecture

```
                          MCP Client (Claude Code)
                                  |
                            stdio (JSON-RPC)
                                  |
                          +-------v--------+
                          |  mcp_server.py |  3 tools: read_wiki_structure
                          |   (lowlevel)   |           read_wiki_contents
                          +---+----+---+---+           ask_question
                              |    |   |
           +------------------+    |   +------------------+
           |                       |                      |
   +-------v-------+      +-------v-------+      +-------v-------+
   | indexer.py     |      | retriever.py  |      | generator.py  |
   | get_structure  |      | hybrid_search |      | claude CLI    |
   | list_repos     |      +-------+-------+      | subprocess    |
   +-------+-------+              |               +---------------+
           |               +------+------+
    +------+------+        |             |
    |             |   +----v----+   +----v----+
    |  ChromaDB   |   | ChromaDB|   |  BM25   |
    |  (persist)  |   | vector  |   | keyword |
    |             |   | query   |   | query   |
    +-------------+   +---------+   +---------+
```

### Data Flow

```
  First query for a repo:

  ask_question("owner/repo", "How does X work?")
       |
       v
  is_indexed?  --NO-->  fetcher.py  -- streamablehttp_client -->  DeepWiki MCP
       |                    |                                  mcp.deepwiki.com/mcp
       |                    v
       |               chunker.py         Split markdown by page/heading
       |                    |             Paragraph split with overlap for
       |                    v             oversized sections. Code blocks atomic.
       |               indexer.py
       |                 /     \
       |          FastEmbed     BM25Okapi
       |          bge-small     tokenized corpus
       |            |               |
       |         ChromaDB       bm25.pkl
       |         (persist)      chunks.pkl
       |              \         /
       YES             \       /
       |                v     v
       +---------->  retriever.py
                     hybrid_search
                       /       \
                  vector       BM25
                  top-25       top-25
                       \       /
                        RRF(k=60)
                          |
                       top-10 chunks
                          |
                     generator.py
                     claude -p --model sonnet
                          |
                       answer text
```

### On-Disk Layout

```
data/
  owner_repo/
    chromadb/          ChromaDB persistent vector store
    bm25.pkl           Serialized BM25Okapi index
    chunks.pkl         Serialized list[WikiChunk] for reconstruction
```

### Key Modules

| File | Role |
|---|---|
| `run_mcp.py` | Entry point. Configures logging to stderr, runs MCP server. |
| `src/mcp_server.py` | MCP tool definitions + dispatch. Stdio transport. |
| `src/ask.py` | Orchestrator: fetch -> index -> retrieve -> generate. |
| `src/fetcher.py` | Direct MCP client to DeepWiki (`streamablehttp_client`). |
| `src/chunker.py` | Markdown-aware splitting by page/heading with overlap. |
| `src/indexer.py` | Dual index build (ChromaDB + BM25). Embedding via FastEmbed. |
| `src/retriever.py` | Hybrid search with Reciprocal Rank Fusion. |
| `src/generator.py` | Claude CLI subprocess for answer generation. |

## Ad-Hoc CLI Usage

All commands assume you're in the wiki project root and use `uv run`.

### Ask a question (full pipeline)

```bash
cd /home/dev/workspace/wiki

uv run python -c "
from src.ask import ask_question
result = ask_question('pipecat-ai/pipecat', 'How does context aggregation work?')
print(result['answer'])
"
```

First call for a repo fetches from DeepWiki and builds the index (~2-3 min on
CPU). Subsequent calls skip straight to retrieval (~15s total).

### Fetch wiki content only (no indexing)

```bash
uv run python -c "
from src.fetcher import fetch_wiki
text = fetch_wiki('pipecat-ai/pipecat')
print(f'{len(text)} chars fetched')
print(text[:2000])
"
```

### Index already-fetched content

```bash
uv run python -c "
from src.fetcher import fetch_wiki
from src.indexer import build_index
text = fetch_wiki('owner/repo')
n = build_index('owner/repo', text)
print(f'Indexed {n} chunks')
"
```

### Search without generation (retrieval only)

```bash
uv run python -c "
from src.indexer import load_index
from src.retriever import hybrid_search
collection, bm25, chunks = load_index('pipecat-ai/pipecat')
results = hybrid_search('transport options', collection, bm25, chunks, n_results=5)
for r in results:
    print(f'[{r.rrf_score:.4f}] {r.chunk.page} > {r.chunk.heading}')
    print(r.chunk.text[:200])
    print()
"
```

### Get wiki structure for an indexed repo

```bash
uv run python -c "
from src.indexer import get_wiki_structure
structure = get_wiki_structure('pipecat-ai/pipecat')
for page, headings in structure.items():
    print(f'## {page}')
    for h in headings:
        print(f'  - {h}')
"
```

### Run as MCP server (stdio)

```bash
uv run python run_mcp.py
```

Logs go to stderr, MCP JSON-RPC goes over stdout/stdin.

### Run with verbose logging

```bash
uv run python -c "
import logging, sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S', stream=sys.stdout)
from src.ask import ask_question
result = ask_question('owner/repo', 'your question here')
print(result['answer'])
"
```

## Maintenance & Debugging

### List indexed repos

```bash
uv run python -c "
from src.indexer import list_indexed_repos
for r in list_indexed_repos():
    print(r)
"
```

### Inspect index stats

```bash
uv run python -c "
import pickle
from pathlib import Path
from src.indexer import load_index

repo = 'pipecat-ai/pipecat'
collection, bm25, chunks = load_index(repo)
print(f'Repo:           {repo}')
print(f'Total chunks:   {len(chunks)}')
print(f'ChromaDB count: {collection.count()}')
print(f'Unique pages:   {len(set(c.page for c in chunks))}')

# Show chunk size distribution
sizes = [len(c.text) for c in chunks]
print(f'Chunk sizes:    min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}')
"
```

### Re-index a repo (clear and rebuild)

```bash
# Delete existing index
rm -rf data/pipecat-ai_pipecat

# Rebuild
uv run python -c "
from src.ask import ask_question
result = ask_question('pipecat-ai/pipecat', 'test query')
print('Re-indexed successfully')
"
```

### Check ChromaDB health

```bash
uv run python -c "
import chromadb
client = chromadb.PersistentClient(path='data/pipecat-ai_pipecat/chromadb')
col = client.get_collection('wiki')
print(f'Collection: {col.name}')
print(f'Count:      {col.count()}')
print(f'Metadata:   {col.metadata}')
sample = col.peek(3)
for i, doc in enumerate(sample['documents']):
    print(f'\nSample {i}: {doc[:120]}...')
"
```

### Verify DeepWiki connectivity

```bash
uv run python -c "
from src.fetcher import fetch_wiki
try:
    text = fetch_wiki('pipecat-ai/pipecat')
    print(f'OK: {len(text)} chars')
except Exception as e:
    print(f'FAILED: {e}')
"
```

### Run tests

```bash
uv run pytest tests/ -v
```

### Check data directory disk usage

```bash
du -sh data/*/
```

### Tail MCP server logs (when running as registered server)

The MCP server logs to stderr. When registered in Claude Code settings,
stderr goes to the MCP log file:

```bash
tail -f ~/.claude/logs/mcp-*.log 2>/dev/null || echo "No MCP logs found"
```
