# main.py
#!/usr/bin/env python3
import sys
import argparse
from chatbot import Chatbot
from utils import setup_logging
from config import OPENAI_MODEL

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Ava AI Assistant")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--model", type=str, default=OPENAI_MODEL, help="OpenAI model to use")
    return parser.parse_args()

def main():
    """Main entry point for the application"""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(verbose=args.verbose)
    
    # Initialize chatbot with specified model and verbosity
    chatbot = Chatbot(model_name=args.model, verbose=args.verbose)
    
    try:
        # Start the chat session
        chatbot.run()
    except KeyboardInterrupt:
        print("\nExiting due to user interrupt...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Make sure to save memory before exiting
        chatbot.save_memory()
        print("Memory saved. Goodbye!")

if __name__ == "__main__":
    main()