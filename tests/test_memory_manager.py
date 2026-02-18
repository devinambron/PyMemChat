from unittest.mock import Mock

from app.memory_manager import MemoryManager


def test_add_memory_stores_correctly() -> None:
    client = Mock()
    mm = MemoryManager(file_path="ignored.json", api_key="test", mem0_client=client)

    mm.add_memory(user_id="u1", message="hello", role="user")
    client.add.assert_called_with([{"role": "user", "content": "hello"}], user_id="u1")

    client.reset_mock()
    mm.add_memory(user_id="u1", message="hi", role="ai")
    client.add.assert_called_with([{"role": "assistant", "content": "hi"}], user_id="u1")


def test_get_relevant_memory_returns_filtered_results() -> None:
    client = Mock()
    client.search.return_value = {
        "results": [
            {"memory": "User likes coffee"},
            {"memory": "   "},
            {"content": "Fallback content field"},
            "Plain string memory",
            {"unexpected": "shape"},
        ]
    }
    mm = MemoryManager(file_path=None, api_key="test", mem0_client=client)

    memories = mm.get_relevant_memory(user_id="u1", query="coffee")
    assert memories == ["User likes coffee", "Fallback content field", "Plain string memory"]


def test_clear_memory_calls_mem0_delete_all() -> None:
    client = Mock()
    mm = MemoryManager(file_path=None, api_key="test", mem0_client=client)

    mm.clear_memory("u1")
    client.delete_all.assert_called_with(user_id="u1")

