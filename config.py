import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    AI_NAME = "Ava"
    MEMORY_FILE = 'chat_memory.json'
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key")
    MODEL_NAME = "gpt-3.5-turbo"
    TEMPERATURE = 0.7
    MAX_TOKENS = 512