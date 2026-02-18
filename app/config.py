import os


class Config:
    AI_NAME = "Ava"
    MEMORY_FILE = "chat_memory.json"

    USER_ID = os.getenv("PYMEMCHAT_USER_ID", "default")

    BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key")

    MODEL_NAME = os.getenv("PYMEMCHAT_MODEL", "gpt-4o-mini")
    TEMPERATURE = float(os.getenv("PYMEMCHAT_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("PYMEMCHAT_MAX_TOKENS", "512"))

