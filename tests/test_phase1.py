"""Phase 1 smoke tests: chunking and indexing."""

from src.chunker import chunk_wiki
from src.indexer import build_index, is_indexed, load_index, _embed_texts

SAMPLE_WIKI = """# Page: Getting Started
## Installation
Install pipecat with pip:
```bash
pip install pipecat-ai
```

## Quick Start
Create a simple pipeline:
```python
from pipecat import Pipeline
pipeline = Pipeline()
```

# Page: Transport Layer
## WebSocket Transport
The WebSocketTransport class handles real-time communication.

### Configuration
Set up transport with host and port parameters.

## HTTP Transport
For request-response patterns, use HTTPTransport.
"""


def test_chunking():
    chunks = chunk_wiki(SAMPLE_WIKI)
    assert len(chunks) > 0
    assert all(c.page for c in chunks)
    assert all(c.text for c in chunks)
    # Code blocks should not be split
    code_chunks = [c for c in chunks if "```" in c.text]
    for c in code_chunks:
        assert c.text.count("```") % 2 == 0, "Code block split across chunks"


def test_chunk_metadata():
    chunks = chunk_wiki(SAMPLE_WIKI)
    pages = {c.page for c in chunks}
    assert "Getting Started" in pages
    assert "Transport Layer" in pages


def test_chunk_indices_sequential():
    chunks = chunk_wiki(SAMPLE_WIKI)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_index_roundtrip(tmp_path):
    data_dir = str(tmp_path)
    repo = "test/repo"
    assert not is_indexed(repo, data_dir)

    count = build_index(repo, SAMPLE_WIKI, data_dir)
    assert count > 0
    assert is_indexed(repo, data_dir)

    collection, bm25, chunks = load_index(repo, data_dir)
    assert collection.count() == count
    assert len(chunks) == count


def test_chromadb_query_works(tmp_path):
    data_dir = str(tmp_path)
    build_index("test/repo", SAMPLE_WIKI, data_dir)
    collection, _, _ = load_index("test/repo", data_dir)

    query_embedding = _embed_texts(["WebSocket transport"])
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    assert len(results["ids"][0]) > 0
    assert len(results["documents"][0]) > 0


def test_bm25_scores_work(tmp_path):
    data_dir = str(tmp_path)
    build_index("test/repo", SAMPLE_WIKI, data_dir)
    _, bm25, chunks = load_index("test/repo", data_dir)

    scores = bm25.get_scores("websocket transport".split())
    assert len(scores) == len(chunks)
    assert max(scores) > 0  # At least one chunk should match
