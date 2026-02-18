# PyMemChat

A lightweight, self-hosted chatbot with semantic memory — your conversations, your data, no cloud lock-in.

## Features

- **Modern LangChain composition**: Uses `RunnableWithMessageHistory` for in-session history.
- **Semantic memory**: Uses `mem0` to store and retrieve relevant past conversation snippets.
- **Clean CLI UX**: Typer-powered command with Rich output, plus inline chat commands.
- **Provider flexibility**: Works with OpenAI-compatible endpoints via `base_url`.

## Installation

```bash
git clone https://github.com/devinambron/PyMemChat.git
cd PyMemChat

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

Run the chat CLI:

```bash
python -m app.main chat --user myname
```

Options:

- `--verbose / -v`: Show debug output and memory context injections.
- `--user`: Namespaces semantic memory by user id (default: `default`).
- `--clear-memory`: Clears semantic memory for the user before starting.

Inline commands during chat:

- `/memory`: Print the last 5 stored memories for the current user.
- `/clear`: Wipe memory for the current user.

## How Memory Works

Older versions of PyMemChat stored conversation history in a flat JSON file and replayed it verbatim. This modern version uses **mem0 semantic memory**:

- Each user/assistant turn is stored as a memory item.
- On each new user input, PyMemChat performs a **semantic search** against stored memories.
- The most relevant results are injected into the model as a **system message** labeled “Relevant memory context:” above the current session’s chat history.

This keeps prompts focused (only the most relevant history is surfaced) while still preserving long-term context.

## Provider Compatibility

PyMemChat uses the OpenAI SDK + `langchain-openai` and supports OpenAI-compatible providers via `base_url`.

- **OpenAI** (default): `OPENAI_API_BASE=https://api.openai.com/v1`
- **OpenRouter**: set `OPENAI_API_BASE` to the OpenRouter base URL and use your OpenRouter API key.
- **Ollama**: set `OPENAI_API_BASE` to your local OpenAI-compatible endpoint (e.g., an OpenAI-compatible proxy in front of Ollama).

Environment variables:

- `OPENAI_API_KEY`: API key for the provider.
- `OPENAI_API_BASE`: Base URL for the provider (defaults to OpenAI).
- `PYMEMCHAT_USER_ID`: Default user id if not provided via `--user`.

## Contributing

PRs welcome — please include a short test plan and run `pytest`.