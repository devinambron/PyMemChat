# PyMemChat

PyMemChat is an open-source chatbot application that utilizes memory management to enhance user interactions. The chatbot remembers previous conversations, allowing for a more personalized and context-aware experience. It leverages the OpenAI API to generate responses based on user input.

## Features

- Memory management to retain conversation history.
- Integration with OpenAI's GPT model for generating responses.
- Verbose logging for debugging and monitoring.

## File Structure

```
PyMemChat/
│
├── chatbot.py          # Main chatbot logic and interaction handling.
├── config.py           # Configuration settings, including API keys and model parameters.
├── exceptions.py       # Custom exception classes for error handling.
├── main.py             # Entry point for running the application.
├── memory_manager.py    # Handles loading and saving conversation memory.
├── utils.py            # Utility functions for logging and processing data.
└── requirements.txt    # List of dependencies for the project.
```

## Installation

To set up a virtual environment for this project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/PyMemChat.git
   cd PyMemChat
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv pymemchat-env
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     pymemchat-env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source pymemchat-env/bin/activate
     ```

4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

To run the chatbot, use the following command:

```bash
python main.py -v
```

The `-v` flag enables verbose logging for debugging purposes.

## How It Works

1. **Initialization**: The `main.py` script initializes the application, sets up logging, and creates an instance of the `Chatbot` class.

2. **Chatbot Logic**: The `chatbot.py` file contains the core logic for handling user interactions. It manages memory through the `MemoryManager` class and generates responses using the OpenAI API.

3. **Memory Management**: The `memory_manager.py` file is responsible for loading and saving conversation history to a JSON file, allowing the chatbot to remember past interactions.

4. **Utilities**: The `utils.py` file provides helper functions for logging and processing user input and memory data.

5. **Configuration**: The `config.py` file holds configuration settings, including API keys and model parameters.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# PyMemChat Test Suite

This directory contains comprehensive tests for the PyMemChat chatbot application, focusing on memory persistence and retrieval across chat sessions.

## Test Components

The test suite consists of the following files:

1. `test_chatbot.py` - Tests for the Chatbot class functionality
2. `test_memory_manager.py` - Tests for the MemoryManager class functionality
3. `run_tests.py` - A script to run all tests with detailed output

## Test Coverage

The tests cover the following scenarios:

- Basic conversation functionality
- Memory persistence within a single session
- Memory persistence across multiple sessions
- Entity recognition and recall
- Context retrieval from previous conversations
- Various memory components (summary, entity memory, vector store)

## Running the Tests

### Prerequisites

Make sure you have installed all the required dependencies and have set up the virtual environment as per the main project instructions.

### Running All Tests

To run all tests, use the `run_tests.py` script:

```bash
python run_tests.py
```

This will run all test cases with detailed output.

### Running Individual Test Files

You can also run individual test files:

```bash
python -m unittest test_chatbot.py
python -m unittest test_memory_manager.py
```

### Running Specific Test Cases

To run a specific test case:

```bash
python -m unittest test_chatbot.TestChatbot.test_basic_conversation
python -m unittest test_memory_manager.TestMemoryManager.test_add_message
```

## Test Design

The tests use Python's `unittest` framework and include:

- Mock objects for LLM and API dependencies
- Temporary files for memory storage
- Patching of external dependencies

## Troubleshooting

If you encounter any issues running the tests:

1. Ensure your virtual environment is activated
2. Check that all dependencies are installed
3. Confirm that the main application code is functioning properly
4. Look for any error messages in the test output

## Adding New Tests

When adding new functionality to the chatbot, please add corresponding tests to maintain code coverage.