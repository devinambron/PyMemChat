import argparse
from chatbot import Chatbot
from utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="PyMemChat - Chatbot with memory.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    chatbot = Chatbot()
    chatbot.run()

if __name__ == "__main__":
    main()