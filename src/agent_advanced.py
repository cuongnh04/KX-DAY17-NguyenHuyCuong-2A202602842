from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # TODO: optionally initialize a real LangChain/LangGraph agent.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        updates = extract_profile_updates(message)
        profile = self.profile_store.read_text(user_id)
        for key, value in updates.items():
            heading = key.replace("_", " ").title()
            pattern = rf"(?m)^- {re.escape(heading)}:.*$"
            line = f"- {heading}: {value}"
            if re.search(pattern, profile):
                profile = re.sub(pattern, line, profile)
            else:
                profile += ("" if profile.endswith("\n") else "\n") + line + "\n"
        if updates:
            self.profile_store.write_text(user_id, profile)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        generated = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + generated
        return {"response": response, "token_usage": generated, "prompt_tokens": prompt_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        context = self.compact_memory.context(thread_id)
        content = self.profile_store.read_text(user_id)
        content += str(context.get("summary", "")) + str(context.get("messages", []))
        return estimate_tokens(content)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        profile = self.profile_store.read_text(user_id)
        facts = {}
        for line in profile.splitlines():
            match = re.match(r"- ([^:]+):\s*(.*)", line)
            if match:
                facts[match.group(1).lower()] = match.group(2).strip()
        lower = message.lower()
        requested = []
        if "tên" in lower: requested.append(("name", "tên"))
        if "nghề" in lower or "làm gì" in lower: requested.append(("profession", "nghề nghiệp"))
        if "ở đâu" in lower or "nơi ở" in lower or "ở hiện" in lower: requested.append(("location", "nơi ở"))
        if "style" in lower or "trả lời" in lower: requested.append(("response_style", "style"))
        if "đồ uống" in lower or "uống" in lower: requested.append(("favorite_drink", "đồ uống"))
        if "món ăn" in lower: requested.append(("favorite_food", "món ăn"))
        if "python" in lower or "ai" in lower: requested.append(("interests", "mối quan tâm"))
        if "corgi" in lower or "bơ" in lower: requested.append(("pet", "thú cưng"))
        if requested:
            values = [f"{label}: {facts[key]}" for key, label in requested if key in facts]
            return "Mình nhớ: " + "; ".join(values) if values else "Mình chưa có thông tin đó."
        return "Mình đã ghi nhận thông tin này và sẽ giữ lại cho các cuộc trò chuyện sau."

    def _maybe_build_langchain_agent(self):
        """Student TODO: wire a live agent with tools and compact middleware.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        return None
