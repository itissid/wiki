"""Phase 1 playground: Fetch, chunk, and index a wiki."""

from src.fetcher import fetch_wiki
from src.chunker import chunk_wiki
from src.indexer import build_index, is_indexed, load_index

REPO = "pipecat-ai/pipecat"
DATA_DIR = "./data"  # local for dev, /mnt/ServiceAndDataPool/projects/wiki in prod


def main():
    if is_indexed(REPO, DATA_DIR):
        print(f"Already indexed: {REPO}")
        collection, bm25, chunks = load_index(REPO, DATA_DIR)
        print(f"  Chunks: {len(chunks)}")
        print(f"  ChromaDB docs: {collection.count()}")
        return

    print(f"Fetching wiki for {REPO}...")
    wiki_text = fetch_wiki(REPO)
    print(f"  Wiki length: {len(wiki_text)} chars")

    print("Chunking...")
    chunks = chunk_wiki(wiki_text)
    print(f"  Chunks: {len(chunks)}")
    print(f"  Sample: {chunks[0].page} / {chunks[0].heading} ({len(chunks[0].text)} chars)")

    print("Indexing...")
    count = build_index(REPO, wiki_text, DATA_DIR)
    print(f"  Indexed {count} chunks into ChromaDB + BM25")


if __name__ == "__main__":
    main()
