"""Stream the text-to-SQL Deep Agent response in the terminal."""

import argparse
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from agent import (
    DEFAULT_MODELS,
    ModelProvider,
    _build_trace_tags,
    create_sql_deep_agent,
)

console = Console()


@dataclass
class StreamAccumulator:
    """Collect provider-neutral LangChain message chunks and final graph state."""

    current_text: str = ""
    next_text_starts_message: bool = False
    final_message: AIMessage | None = None

    def consume_message_event(self, data: object) -> str | None:
        """Consume the message chunk contained in a LangGraph message event."""
        if not isinstance(data, tuple) or not data:
            return None
        chunk = data[0]
        if not isinstance(chunk, AIMessageChunk):
            return None

        text = str(chunk.text)
        if text:
            if self.next_text_starts_message:
                self.current_text = ""
            self.current_text += text
            self.next_text_starts_message = False
        if chunk.chunk_position == "last":
            self.next_text_starts_message = True
        return self.current_text if text else None

    def consume_values_event(self, data: object) -> None:
        """Remember the latest complete AI message from the graph state."""
        if not isinstance(data, dict):
            return
        messages = data.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, str):
            return
        if messages and isinstance(messages[-1], AIMessage):
            self.final_message = messages[-1]

    def answer(self) -> str:
        """Return the complete final answer, falling back to streamed text."""
        if self.final_message is not None:
            return str(self.final_message.text)
        if self.current_text:
            return self.current_text
        msg = "The agent completed without returning a text answer."
        raise RuntimeError(msg)


def _answer_panel(answer: str, *, complete: bool = False) -> Panel:
    """Build the live answer panel without interpreting model text as Rich markup."""
    body = Text(answer) if answer else Text("Waiting for model output...", style="dim")
    title = "Answer" if complete else "Live model output"
    return Panel(body, title=f"[bold green]{title}[/bold green]", border_style="green")


def _stream_answer(
    agent: CompiledStateGraph,
    question: str,
    config: RunnableConfig,
) -> str:
    """Stream model text while retaining the graph's exact final answer."""
    accumulator = StreamAccumulator()
    raw_events = agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode=["messages", "values"],
        version="v1",
    )
    events = cast(Iterator[tuple[str, object]], raw_events)

    with Live(_answer_panel(""), console=console, refresh_per_second=12) as live:
        for mode, data in events:
            if mode == "messages":
                if text := accumulator.consume_message_event(data):
                    live.update(_answer_panel(text))
            elif mode == "values":
                accumulator.consume_values_event(data)
        answer = accumulator.answer()
        live.update(_answer_panel(answer, complete=True), refresh=True)
    return answer


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the streaming example."""
    parser = argparse.ArgumentParser(
        description="Streaming Text-to-SQL Deep Agent",
    )
    parser.add_argument("question", help="Natural language database question")
    parser.add_argument(
        "--provider",
        choices=tuple(DEFAULT_MODELS),
        help="Model provider (inferred from --model when omitted)",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        help="Provider-specific model name",
    )
    return parser.parse_args()


def main() -> None:
    """Run the streaming text-to-SQL CLI."""
    args = _parse_args()
    console.print(
        Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan")
    )
    console.print("\n[dim]Creating SQL Deep Agent...[/dim]")

    provider = cast(ModelProvider | None, args.provider)
    try:
        agent = create_sql_deep_agent(provider=provider, model_name=args.model_name)
        tags = _build_trace_tags(provider, args.model_name)
        console.print("[dim]Processing query...[/dim]\n")
        _stream_answer(agent, args.question, {"tags": tags})
    except Exception as error:  # noqa: BLE001  # CLI boundary reports provider errors.
        console.print(
            Panel(f"[bold red]Error:[/bold red]\n\n{error!s}", border_style="red")
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
