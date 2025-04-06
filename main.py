# main.py
import sys
import argparse
from chatbot import Chatbot
from utils import setup_logging

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PyMemChat - A chatbot with enhanced memory abilities")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def main():
    """Main function to run the chatbot"""
    args = parse_args()
    setup_logging(args.verbose)
    
    # Initialize and run the chatbot
    chatbot = Chatbot()
    chatbot.run()

if __name__ == "__main__":
    main()