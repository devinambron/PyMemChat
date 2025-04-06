# main.py
import sys
import logging
import argparse
from chatbot import Chatbot
from utils import setup_logging, sanitize_user_input

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the memory-enhanced chatbot")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def main():
    """Main entry point for the chatbot application"""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    verbose = args.verbose
    setup_logging(verbose)
    
    logger = logging.getLogger("main")
    logger.info("Starting chatbot")
    
    try:
        # Initialize the chatbot
        chatbot = Chatbot()
        
        # Print welcome message
        print(f"\n{chatbot.config.AI_NAME} is ready to chat! Type 'exit' to end the conversation.\n")
        
        # Main chat loop
        while True:
            # Get user input
            user_input = input("You: ")
            user_input = sanitize_user_input(user_input)
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "bye"]:
                print(f"\n{chatbot.config.AI_NAME}: Goodbye! Have a great day.")
                logger.info("User has chosen to exit the chat.")
                break
                
            # Get response from the chatbot
            try:
                print(f"{chatbot.config.AI_NAME}: ", end="", flush=True)
                response = chatbot.generate_response(user_input)
                print(response)
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                print(f"\nI'm sorry, I encountered an error: {str(e)}")
        
        # Save chat memory before exiting
        chatbot.save_memory()
        logger.info("Memory saved successfully. Chat ended.")
        
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print(f"\nCritical error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())