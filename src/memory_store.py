from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


def estimate_tokens(text: str) -> int:
    cleaned = " ".join(text.split())
    return 0 if not cleaned else max(1, (len(cleaned) + 3) // 4)


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", user_id).strip("_") or "anonymous"
        return self.root_dir / cleaned / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else "# User profile\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0


def extract_profile_updates(message: str) -> dict[str, str]:
    text = " ".join(message.strip().split())
    if not text or text.endswith(("?", "？")):
        return {}
    patterns = [
        ("name", r"(?:mình|tôi)\s+tên\s+là\s+([^,.!?。]+)"),
        ("location", r"(?:hiện\s+)?(?:mình|tôi)\s+(?:đang\s+)?ở\s+([^,.!?。]+)"),
        ("profession", r"(?:hiện\s+tại\s+)?(?:mình|tôi)\s+đang\s+làm\s+(?!việc)([^,.!?。]+)"),
        ("profession", r"(?:giờ\s+)?(?:(?:mình|tôi)\s+)?chuyển\s+sang\s+([^,.!?。]+)"),
        ("response_style", r"(?:mình|tôi)\s+(?:muốn|thích)\s+(?:bạn\s+)?trả\s+lời\s+([^,.!?。]+)"),
        ("favorite_drink", r"(?:đồ uống yêu thích|thức uống yêu thích)\s+(?:của mình\s+)?(?:là\s+)?([^,.!?。]+)"),
        ("favorite_food", r"món ăn yêu thích\s+(?:là\s+)?([^,.!?。]+)"),
        ("interests", r"(?:mình|tôi)\s+thích\s+([^,.!?。]+)"),
    ]
    updates: dict[str, str] = {}
    for key, pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = re.split(r"\s+(?:và|nhưng|để|chứ)\s+", value, maxsplit=1, flags=re.I)[0].strip()
            if key == "favorite_drink" and not re.search(r"cà phê|trà|nước|sinh tố|đồ uống", value, re.I):
                continue
            if key == "interests" and re.search(r"cà phê|trà|nước|sinh tố", value, re.I):
                continue
            updates[key] = value
    for match in re.finditer(r"(?:mình|tôi)\s+thích\s+([^,.!?。]+)", text, re.I):
        value = match.group(1).strip()
        if re.search(r"cà phê|trà|nước|sinh tố", value, re.I):
            updates["favorite_drink"] = value
    pet = re.search(r"(?:nuôi|con)\s+(?:một\s+)?(?:bé\s+)?(corgi(?:\s+tên\s+[A-Za-zÀ-ỹ]+)?)", text, re.I)
    if pet:
        updates["pet"] = pet.group(1).strip()
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    return " | ".join(f"{item['role']}: {item['content']}" for item in messages[-max_items:])


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        item = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = item["messages"]
        messages.append({"role": role, "content": content})
        if estimate_tokens(str(messages) + str(item["summary"])) > self.threshold_tokens:
            older = messages[:-self.keep_messages]
            item["summary"] = " ".join(filter(None, [item["summary"], summarize_messages(older)]))
            item["messages"] = messages[-self.keep_messages:]
            item["compactions"] = int(item["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
