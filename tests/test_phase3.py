"""Phase 3 smoke tests: Claude CLI and end-to-end Q&A."""

import json
import subprocess


def test_claude_cli_available():
    result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Claude Code" in result.stdout


def test_claude_cli_responds():
    """Verify claude -p works with basic prompt."""
    result = subprocess.run(
        [
            "claude", "-p", "Say 'hello' and nothing else.",
            "--no-session-persistence",
            "--max-turns", "1", "--output-format", "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "hello" in data["result"].lower()


def test_claude_cli_uses_piped_context():
    """Verify claude uses stdin context."""
    result = subprocess.run(
        [
            "claude", "-p", "What is the secret word in the context?",
            "--no-session-persistence",
            "--max-turns", "1", "--output-format", "json",
            "--system-prompt", "Answer based only on the provided context. Be brief.",
        ],
        input="The secret word is PINEAPPLE.",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "pineapple" in data["result"].lower()
