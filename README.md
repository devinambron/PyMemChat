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