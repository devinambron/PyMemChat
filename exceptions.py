# exceptions.py
import logging

logger = logging.getLogger(__name__)

class MemoryLoadError(Exception):
    """Raised when there's an error loading the memory file."""
    def __init__(self, message: str):
        super().__init__(message)
        logger.error(message)

class MemorySaveError(Exception):
    """Raised when there's an error saving the memory file."""
    def __init__(self, message: str):
        super().__init__(message)
        logger.error(message)

class APICallError(Exception):
    """Raised when there's an error calling the OpenAI API."""
    def __init__(self, message: str):
        super().__init__(message)
        logger.error(message)