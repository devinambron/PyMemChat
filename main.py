# main.py
#!/usr/bin/env python3
import sys
import argparse
import os
import shutil
from chatbot import Chatbot, start_chat
from utils import setup_logging
from config import OPENAI_MODEL
from dotenv import load_dotenv
from memory_manager import MemoryManager
from langchain_openai import ChatOpenAI

def clear_memory(memory_file="chat_memory.json", vector_store_dir="vector_store"):
    """Clears all memory by deleting memory file and vector store directory."""
    print(f"Attempting to clear memory...")
    
    # Delete memory file
    if os.path.exists(memory_file):
        os.remove(memory_file)
        print(f"Memory file '{memory_file}' deleted.")
    else:
        print(f"Memory file '{memory_file}' not found, skipping deletion.")
    
    # Delete vector store directory
    if os.path.exists(vector_store_dir):
        shutil.rmtree(vector_store_dir)
        print(f"Vector store directory '{vector_store_dir}' deleted.")
    else:
        print(f"Vector store directory '{vector_store_dir}' not found, skipping deletion.")
    
    print("Memory clearing process complete.")

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Ava AI Assistant")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--model", type=str, default=OPENAI_MODEL, help="OpenAI model to use")
    parser.add_argument("--user", type=str, default="User", help="Your name")
    parser.add_argument("--clear-memory", action="store_true", help="Clear memory before starting")
    return parser.parse_args()

def main():
    """Main entry point for the application"""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(verbose=args.verbose)
    
    # Clear memory if requested
    if args.clear_memory:
        clear_memory()
        print("Memory cleared successfully.")
    
    # Instead of initializing directly, use the start_chat function which handles
    # memory loading and creates a proper interactive session
    try:
        # Call the start_chat function from the chatbot module
        # This handles memory management, chatbot initialization, and the chat loop
        print(f"Starting chat with model: {args.model}")
        print(f"User: {args.user}")
        print(f"Streaming mode: enabled (default)")
        
        # Start the interactive chat session
        start_chat(
            verbose=args.verbose,
            model_name=args.model,
            user_name=args.user
        )
        
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()