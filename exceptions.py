class MemoryLoadError(Exception):
    """Raised when there's an error loading the memory file."""
    pass

class MemorySaveError(Exception):
    """Raised when there's an error saving the memory file."""
    pass

class APICallError(Exception):
    """Raised when there's an error calling the OpenAI API."""
    pass