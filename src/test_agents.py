from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def make_config(tmp_path: Path):
    """Student TODO: build an isolated config for tests."""

    # Hint:
    # - point `state_dir` into tmp_path
    # - reduce compact threshold so compaction happens quickly in tests
    config = load_config(tmp_path)
    config.state_dir = tmp_path / "state"
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.compact_threshold_tokens = 30
    config.compact_keep_messages = 2
    return config


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Student TODO: verify `User.md` can be created, updated, and edited."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    agent.profile_store.write_text("u1", "# User profile\n- Name: An\n")
    assert "Name: An" in agent.profile_store.read_text("u1")
    assert agent.profile_store.edit_text("u1", "Name: An", "Name: Bình")
    assert "Name: Bình" in agent.profile_store.read_text("u1")


def test_compact_trigger(tmp_path: Path) -> None:
    """Student TODO: verify long threads trigger compaction."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    for i in range(6):
        agent.reply("u1", "thread", f"Tin nhắn dài số {i} " + "x" * 80)
    assert agent.compaction_count("thread") > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Student TODO: verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    advanced.reply("u1", "first", "Mình tên là An.")
    baseline.reply("u1", "first", "Mình tên là An.")
    assert "An" in advanced.reply("u1", "new", "Mình tên gì?")["response"]
    assert "An" not in baseline.reply("u1", "new", "Mình tên gì?")["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Student TODO: compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for i in range(12):
        message = f"message {i} " + "z" * 100
        baseline.reply("u1", "long", message)
        advanced.reply("u1", "long", message)
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
