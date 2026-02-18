from __future__ import annotations

import os
from typing import Any, Iterable

from mem0 import Memory


class MemoryManager:
    """
    Semantic memory manager backed by mem0.

    The `file_path` argument is kept for backward compatibility but is ignored;
    mem0 persists internally (SQLite by default).
    """

    def __init__(
        self,
        file_path: str | None,
        *,
        api_key: str,
        mem0_client: Any | None = None,
    ) -> None:
        self.file_path = file_path
        self._client = mem0_client or self._initialize_mem0(api_key=api_key)

    @staticmethod
    def _initialize_mem0(*, api_key: str) -> Any:
        os.environ["OPENAI_API_KEY"] = api_key
        return Memory()

    @staticmethod
    def _normalize_results(result: Any) -> list[Any]:
        if result is None:
            return []
        if isinstance(result, dict):
            results = result.get("results")
            if isinstance(results, list):
                return results
            return []
        if isinstance(result, list):
            return result
        return []

    @staticmethod
    def _extract_memory_text(item: Any) -> str | None:
        if isinstance(item, str):
            return item.strip() or None
        if isinstance(item, dict):
            mem = item.get("memory") or item.get("content") or item.get("text")
            if isinstance(mem, str) and mem.strip():
                return mem.strip()
        return None

    def add_memory(self, user_id: str, message: str, role: str) -> None:
        role_normalized = "assistant" if role.lower() in {"ai", "assistant"} else "user"
        payload = [{"role": role_normalized, "content": message}]
        self._client.add(payload, user_id=user_id)

    def get_relevant_memory(self, user_id: str, query: str) -> list[str]:
        result = self._client.search(query, user_id=user_id)
        items = self._normalize_results(result)
        memories: list[str] = []
        for item in items:
            text = self._extract_memory_text(item)
            if text:
                memories.append(text)
        return memories

    def get_last_memories(self, user_id: str, *, limit: int = 5) -> list[str]:
        result = self._client.get_all(user_id=user_id, limit=limit)
        items = self._normalize_results(result)
        memories: list[str] = []
        for item in items:
            text = self._extract_memory_text(item)
            if text:
                memories.append(text)
        return memories[:limit]

    def clear_memory(self, user_id: str) -> None:
        self._client.delete_all(user_id=user_id)

