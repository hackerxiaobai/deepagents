import argparse
import os
import sys
from typing import Literal, cast

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()

ModelProvider = Literal["anthropic", "openai"]
DEFAULT_MODELS: dict[ModelProvider, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5.6-terra",
}


def _create_model(
    provider: ModelProvider, model_name: str | None = None
) -> BaseChatModel:
    """Create a chat model for the requested provider.

    Args:
        provider: Model provider to use.
        model_name: Provider-specific model name. Uses the provider default when omitted.

    Returns:
        A configured LangChain chat model.

    Raises:
        ValueError: If `provider` is unsupported.
    """
    resolved_model = model_name or DEFAULT_MODELS.get(provider)
    if resolved_model is None:
        msg = f"Unsupported model provider: {provider}"
        raise ValueError(msg)

    if provider == "anthropic":
        return ChatAnthropic(model_name=resolved_model, temperature=0)
    if provider == "openai":
        return ChatOpenAI(model=resolved_model, use_responses_api=True,)

    msg = f"Unsupported model provider: {provider}"
    raise ValueError(msg)


def create_sql_deep_agent(
    *,
    provider: ModelProvider = "anthropic",
    model_name: str | None = None,
) -> CompiledStateGraph:
    """Create a text-to-SQL Deep Agent.

    Args:
        provider: Model provider to use.
        model_name: Provider-specific model name. Uses the provider default when omitted.

    Returns:
        A compiled text-to-SQL Deep Agent graph.
    """

    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Connect to Chinook database
    db_path = os.path.join(base_dir, "chinook.db")
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)

    # Use the same model for the SQL toolkit and the Deep Agent.
    model = _create_model(provider, model_name)

    # Create SQL toolkit and get tools
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    # Create the Deep Agent with all parameters
    agent = create_deep_agent(
        model=model,
        memory=["./AGENTS.md"],  # Agent identity and general instructions
        skills=[
            "./skills/"
        ],  # Specialized workflows (query-writing, schema-exploration)
        tools=sql_tools,  # SQL database tools
        subagents=[],  # No subagents needed
        backend=FilesystemBackend(
            root_dir=base_dir, virtual_mode=True
        ),  # Persistent file storage
    )

    return agent


def main() -> None:
    """Run the text-to-SQL Deep Agent CLI."""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Deep Agent powered by LangChain Deep Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "What are the top 5 best-selling artists?"
  python agent.py --provider openai "What are the top 5 best-selling artists?"
  python agent.py --provider openai --model gpt-5.6-sol "How many customers are from Canada?"
  python agent.py "Which employee generated the most revenue by country?"
        """,
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question to answer using the Chinook database",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(DEFAULT_MODELS),
        default="anthropic",
        help="Model provider to use (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        help="Provider-specific model name (default: provider's recommended model)",
    )

    args = parser.parse_args()

    # Display the question
    console.print(
        Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan")
    )
    console.print()

    # Create the agent
    console.print("[dim]Creating SQL Deep Agent...[/dim]")
    provider = cast(ModelProvider, args.provider)
    agent = create_sql_deep_agent(provider=provider, model_name=args.model_name)

    # Invoke the agent
    console.print("[dim]Processing query...[/dim]\n")

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.question}]}
        )

        # Extract and display the final answer
        final_message = result["messages"][-1]
        answer = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        console.print(
            Panel(f"[bold green]Answer:[/bold green]\n\n{answer}", border_style="green")
        )

    except Exception as error:  # noqa: BLE001  # CLI boundary reports provider errors.
        console.print(
            Panel(f"[bold red]Error:[/bold red]\n\n{error!s}", border_style="red")
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
