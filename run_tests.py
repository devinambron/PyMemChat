#!/usr/bin/env python
import unittest
import sys
import os
from unittest import TextTestRunner, TestLoader
from test_chatbot import TestChatbot

def run_tests():
    """Run all tests with detailed output"""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Create a test suite
    loader = TestLoader()
    suite = loader.loadTestsFromTestCase(TestChatbot)
    
    # Run the tests with verbosity=2 for detailed output
    runner = TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests()) 