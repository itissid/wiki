"""Phase 3 playground: End-to-end question answering."""

from src.ask import ask_question
from src.indexer import is_indexed

REPO = "pipecat-ai/pipecat"
DATA_DIR = "./data"

QUESTIONS = [
    "How do I install pipecat?",
    "What transport options does pipecat support?",
    "How do I create a basic voice bot with pipecat?",
]


def main():
    if not is_indexed(REPO, DATA_DIR):
        print(f"Index not found for {REPO}. Run playground_phase1.py first.")
        return

    for q in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        result = ask_question(REPO, q, data_dir=DATA_DIR)
        print(f"A: {result['answer'][:500]}")
        print(f"\n  Retrieval: {result['retrieval_time_ms']:.0f}ms")
        print(f"  Generation: {result['generation_time_ms']:.0f}ms")
        print(f"  Chunks used: {result['chunks_used']}")


if __name__ == "__main__":
    main()
