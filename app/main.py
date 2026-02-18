import logging

import typer
from rich.console import Console
from rich.panel import Panel

from app.chatbot import Chatbot
from app.config import Config
from app.memory_manager import MemoryManager
from app.utils import sanitize_user_input, setup_logging

console = Console()
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="PyMemChat — a lightweight chatbot with semantic memory.")


def _run_chat(
    verbose: bool,
    user: str,
    clear_memory: bool,
) -> None:
    setup_logging(verbose)

    config = Config()

    memory_manager = MemoryManager(
        config.MEMORY_FILE,
        api_key=config.OPENAI_API_KEY,
    )

    mem0_status = "ready"
    try:
        if clear_memory:
            memory_manager.clear_memory(user)
    except Exception as e:
        mem0_status = f"error ({e})"

    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold]PyMemChat[/bold]",
                    f"Model: [bold]{config.MODEL_NAME}[/bold]",
                    f"User: [bold]{user}[/bold]",
                    f"mem0: [bold]{mem0_status}[/bold]",
                ]
            ),
            border_style="cyan",
        )
    )

    chatbot = Chatbot(config=config, memory_manager=memory_manager)

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting.[/dim]")
            return

        user_input = sanitize_user_input(user_input)
        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            console.print("[dim]Goodbye.[/dim]")
            return

        if user_input == "/clear":
            memory_manager.clear_memory(user)
            console.print("[dim]Memory cleared.[/dim]")
            continue

        if user_input == "/memory":
            memories = memory_manager.get_last_memories(user, limit=5)
            if not memories:
                console.print("[dim]No stored memories.[/dim]")
                continue
            console.print("[dim]Last 5 memories:[/dim]")
            for m in memories:
                console.print(f"[dim]- {m}[/dim]")
            continue

        def verbose_context_printer(memory_context: str) -> None:
            console.print(f"[dim italic]Relevant memory context:\\n{memory_context}[/dim italic]")
            console.print(f"[bold green]{config.AI_NAME}:[/bold green] ", end="")

        if not verbose:
            console.print(f"[bold green]{config.AI_NAME}:[/bold green] ", end="")

        try:
            chatbot.generate_response(
                user_input,
                user_id=user,
                session_id=user,
                on_memory_context=verbose_context_printer if verbose else None,
            )
            console.print()
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    user: str = typer.Option("default", "--user", help="User id for memory namespacing."),
    clear_memory: bool = typer.Option(False, "--clear-memory", help="Clear memory before starting."),
) -> None:
    if ctx.invoked_subcommand is None:
        _run_chat(verbose=verbose, user=user, clear_memory=clear_memory)


if __name__ == "__main__":
    app()

