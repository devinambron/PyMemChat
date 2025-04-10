# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# AI Configuration
AI_NAME = "Ava"
MEMORY_FILE = 'chat_memory.json'

# OpenAI API Configuration
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in the environment variables.")

# Model Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
TEMPERATURE = 0.7
MAX_TOKENS = 512

# Deprecated Config class - maintained for backwards compatibility
class Config:
    AI_NAME = AI_NAME
    MEMORY_FILE = MEMORY_FILE
    OPENAI_API_BASE = OPENAI_API_BASE
    OPENAI_API_KEY = OPENAI_API_KEY
    MODEL_NAME = OPENAI_MODEL
    TEMPERATURE = TEMPERATURE
    MAX_TOKENS = MAX_TOKENS