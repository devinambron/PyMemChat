# debug_openai.py
import os
from dotenv import load_dotenv
from openai import OpenAI
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

def main():
    # Get API key and base URL from environment
    api_key = os.getenv("OPENAI_API_KEY", "your_api_key")
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:1234/v1")
    
    logger.debug(f"Using API Base: {api_base}")
    
    # Initialize the client with only supported parameters
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
    )
    
    # Test the client with a simple call
    try:
        models = client.models.list()
        logger.debug(f"Available models: {models}")
        print("OpenAI client initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {e}")
        print(f"Failed to initialize OpenAI client. Error: {e}")
        
if __name__ == "__main__":
    main() 