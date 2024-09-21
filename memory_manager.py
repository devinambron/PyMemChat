import json
import logging
from typing import List, Dict
from langchain.schema.messages import AIMessage, HumanMessage
from exceptions import MemoryLoadError, MemorySaveError

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_memory(self) -> List[Dict[str, str]]:
        logger.debug(f"Loading memory from file: {self.file_path}")
        try:
            with open(self.file_path, 'r') as f:
                memory = json.load(f)
            logger.debug(f"Memory loaded successfully: {memory}")
            return memory
        except FileNotFoundError:
            logger.warning("Memory file not found, initializing empty memory.")
            return []
        except json.JSONDecodeError as e:
            raise MemoryLoadError(f"Error decoding memory file: {e}")

    def save_memory(self, messages: List[AIMessage | HumanMessage]) -> None:
        logger.debug(f"Saving memory to file: {self.file_path}")
        try:
            serializable_memory = [
                {"role": "user", "content": msg.content} if isinstance(msg, HumanMessage) else
                {"role": "ai", "content": msg.content}
                for msg in messages
            ]
            with open(self.file_path, 'w') as f:
                json.dump(serializable_memory, f)
            logger.debug("Memory saved successfully.")
        except IOError as e:
            raise MemorySaveError(f"Error saving memory to file: {e}")