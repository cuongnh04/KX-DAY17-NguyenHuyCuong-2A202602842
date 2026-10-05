from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Student TODO: read JSON conversations from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Student TODO: return 0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 1.0
    hits = sum(1 for item in expected if item.casefold() in answer.casefold())
    return hits / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Student TODO: add a lightweight quality score for offline mode."""

    return recall_points(answer, expected)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Student TODO: evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    before = agent.memory_file_size(conversations[0]["user_id"]) if hasattr(agent, "memory_file_size") else 0
    answers = []
    expected_all = []
    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = conversation["id"]
        for turn in conversation["turns"]:
            result = agent.reply(user_id, thread_id, turn)
            answers.append(result["response"])
        for index, question in enumerate(conversation.get("recall_questions", [])):
            result = agent.reply(user_id, f"{thread_id}-recall-{index}", question["question"])
            answers.append(result["response"])
            expected_all.append(question["expected_contains"])
    recall = sum(recall_points(a, e) for a, e in zip(answers[-len(expected_all):], expected_all)) / len(expected_all) if expected_all else 0
    quality = sum(heuristic_quality(a, e) for a, e in zip(answers[-len(expected_all):], expected_all)) / len(expected_all) if expected_all else 0
    after = agent.memory_file_size(conversations[0]["user_id"]) if hasattr(agent, "memory_file_size") else 0
    thread_ids = [c["id"] for c in conversations]
    return BenchmarkRow(agent_name, sum(agent.token_usage(t) for t in thread_ids),
                        sum(agent.prompt_token_usage(t) for t in thread_ids), recall,
                        quality, after,
                        sum(agent.compaction_count(t) for t in thread_ids))


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Student TODO: print a markdown table or tabulated output."""

    headers = ["Agent", "Agent tokens only", "Prompt tokens processed",
               "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append(f"| {row.agent_name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} | "
                     f"{row.recall_score:.2f} | {row.response_quality:.2f} | "
                     f"{row.memory_growth_bytes} | {row.compactions} |")
    return "\n".join(lines)


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    config = load_config(Path(__file__).resolve().parent.parent)

    for title, filename in [("Standard Benchmark", "conversations.json"),
                            ("Long-Context Stress Benchmark", "advanced_long_context.json")]:
        conversations = load_conversations(config.data_dir / filename)
        rows = [
            run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), conversations, config),
            run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config),
        ]
        print(f"\n## {title}\n\n{format_rows(rows)}")


if __name__ == "__main__":
    main()
