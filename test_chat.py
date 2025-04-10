#!/usr/bin/env python3
"""
Script for testing the Chatbot functionality via command-line.

Allows sending a sequence of inputs to simulate a conversation and triggering
memory clearing for clean testing cycles.
"""
import argparse
import os
import shutil
import logging
import uuid # Import uuid for session IDs
import sys
import time
from typing import List, Optional
from chatbot import Chatbot
from memory_manager import MemoryManager
from exceptions import MemoryLoadError, MemorySaveError
from utils import setup_logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Setup logging for the test script
setup_logging(verbose=True)
logger = logging.getLogger(__name__)

# --- Memory Paths ---
# Ensure these match the paths used in memory_manager.py and vector_store.py
MEMORY_FILE = "chat_memory.json"
VECTOR_STORE_DIR = "vector_store" # Assuming this is the directory name

def clear_memory(memory_file="chat_memory.json", vector_store_dir="vector_store"):
    """Clears all memory by deleting the memory file and vector store directory."""
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

def simulate_conversation(inputs: List[str], verbose: bool = False):
    """Simulates a conversation with the chatbot using a sequence of user inputs."""
    print("Entering simulate_conversation function")
    logging.info("Starting conversation simulation with %d inputs", len(inputs))
    
    user_name = "Devin"  # Test user name
    ai_name = "Ava"
    session_id = "test-session-1"  # Fixed session ID for testing
    memory_file = "chat_memory.json"
    streaming = True # Streaming is now default
    
    try:
        print("Initializing memory manager...")
        # Setup for memory manager
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logging.error("OpenAI API key not found in environment variables.")
            print("ERROR: OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
            return
            
        llm_for_memory = None
        try:
            llm_for_memory = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.0)
            print("LLM for memory initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize LLM for memory: {e}", exc_info=True)
            print(f"Warning: Failed to initialize LLM for memory: {e}")
            # Continue without summarization capabilities
        
        # Initialize memory manager
        memory_manager = MemoryManager(file_path=memory_file, llm=llm_for_memory)
        print("Memory manager created")
        
        # Load existing memory if available
        try:
            memory_manager.load_memory()
            print("Existing memory loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load memory: {e}", exc_info=True)
            print(f"Warning: Failed to load memory: {e}")
            # Continue with empty memory
        
        # Initialize chatbot
        print(f"Initializing chatbot with streaming=True (default)...")
        chatbot = Chatbot(
            memory_manager=memory_manager, 
            user_name=user_name, 
            ai_name=ai_name, 
            verbose=verbose,
            streaming=streaming # Pass True explicitly or rely on default
        )
        print("Chatbot initialized successfully")
        
        # Simulate conversation loop
        print(f"\n--- Starting conversation simulation (Session: {session_id}) ---")
        for i, user_input in enumerate(inputs):
            print(f"\n[Input {i+1}/{len(inputs)}] {user_name}: {user_input}")
            sys.stdout.flush()  # Ensure the input is displayed
            
            try:
                response = chatbot.generate_response(user_input, session_id=session_id)
                # If streaming is enabled, the response is already printed during generation
                if not streaming:
                     # This block should ideally not be reached if streaming=True
                    print(f"{ai_name}: {response}")
                time.sleep(0.5)  # Small delay between exchanges for readability
            except Exception as e:
                logging.error(f"Error in conversation turn {i+1}: {e}", exc_info=True)
                print(f"ERROR in conversation turn {i+1}: {e}")
                # Continue with next input
        
        # End of session: persist memory and create summary
        print("\n--- End of conversation simulation ---")
        try:
            memory_manager.save_memory()
            print("Memory saved successfully")
            
            # Create and store session summary
            session_messages = memory_manager.get_chat_history()
            if session_messages:
                print(f"Creating summary for {len(session_messages)} messages...")
                memory_manager.create_and_store_session_summary(session_messages)
                print("Session summary created and stored.")
            else:
                print("No messages to summarize.")
        except Exception as e:
            logging.error(f"Error during end-of-session processing: {e}", exc_info=True)
            print(f"ERROR during end-of-session processing: {e}")
        
        print("\nConversation simulation completed successfully.")
    
    except KeyboardInterrupt:
        print("\nConversation simulation interrupted by user.")
    except Exception as e:
        logging.error(f"Unexpected error in conversation simulation: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Conversation simulation failed: {e}")

def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Test script for chatbot conversation simulation")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of user inputs for conversation simulation")
    parser.add_argument("--clear-memory", action="store_true", help="Clear all memory before running")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    # parser.add_argument("--streaming", action="store_true", help="Enable streaming mode for the chatbot") # Removed - Streaming is now default
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING # Changed default non-verbose level
    # Pass the adjusted level to setup_logging
    setup_logging(verbose=args.verbose, level=log_level)
    
    # Clear memory if requested
    if args.clear_memory:
        clear_memory()
    
    # Simulate conversation with provided inputs
    # Removed streaming=args.streaming argument
    simulate_conversation(args.inputs, verbose=args.verbose)

if __name__ == "__main__":
    main() 