import argparse
import os
import sys
from collections.abc import Callable
from typing import Literal, cast

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_qwq import ChatQwen
from langchain_xai import ChatXAI
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr
from rich.console import Console
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()

ModelProvider = Literal[
    "anthropic", "openai", "kimi", "glm", "deepseek", "gemini", "qwen", "grok"
]
DEFAULT_MODELS: dict[ModelProvider, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-5.6-terra",
    "kimi": "kimi-k3",
    "glm": "glm-5.3",
    "deepseek": "deepseek-v4-flash",
    "gemini": "gemini-3.7-flash",
    "qwen": "qwen3.8-max",
    "grok": "grok-4.6",
}
MODEL_PREFIXES: tuple[tuple[str, ModelProvider], ...] = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("kimi", "kimi"),
    ("moonshot", "kimi"),
    ("glm", "glm"),
    ("deepseek", "deepseek"),
    ("gemini", "gemini"),
    ("qwen", "qwen"),
    ("qwq", "qwen"),
    ("grok", "grok"),
)


def _require_api_key(name: str, provider: ModelProvider) -> SecretStr:
    """Read a required provider API key from the environment."""
    value = os.getenv(name)
    if value:
        return SecretStr(value)
    msg = f"Set {name} in .env before using the {provider} provider."
    raise ValueError(msg)


def _create_anthropic_model(model_name: str) -> BaseChatModel:
    return ChatAnthropic(model_name=model_name, temperature=0)


def _create_openai_model(model_name: str) -> BaseChatModel:
    return ChatOpenAI(model=model_name, use_responses_api=True)


def _create_kimi_model(model_name: str) -> BaseChatModel:
    return ChatOpenAI(
        model=model_name,
        api_key=_require_api_key("MOONSHOT_API_KEY", "kimi"),
        base_url="https://api.moonshot.ai/v1",
    )


def _create_glm_model(model_name: str) -> BaseChatModel:
    return ChatOpenAI(
        model=model_name,
        api_key=_require_api_key("ZAI_API_KEY", "glm"),
        base_url="https://api.z.ai/api/paas/v4/",
    )


def _create_deepseek_model(model_name: str) -> BaseChatModel:
    return ChatDeepSeek(model=model_name)


def _create_gemini_model(model_name: str) -> BaseChatModel:
    return ChatGoogleGenerativeAI(model=model_name)


def _create_qwen_model(model_name: str) -> BaseChatModel:
    return ChatQwen(model=model_name)


def _create_grok_model(model_name: str) -> BaseChatModel:
    return ChatXAI(model=model_name)


MODEL_FACTORIES: dict[ModelProvider, Callable[[str], BaseChatModel]] = {
    "anthropic": _create_anthropic_model,
    "openai": _create_openai_model,
    "kimi": _create_kimi_model,
    "glm": _create_glm_model,
    "deepseek": _create_deepseek_model,
    "gemini": _create_gemini_model,
    "qwen": _create_qwen_model,
    "grok": _create_grok_model,
}


def _infer_provider(model_name: str) -> ModelProvider:
    """Infer a provider from a model name prefix."""
    normalized_name = model_name.lower()
    for prefix, provider in MODEL_PREFIXES:
        if normalized_name.startswith(prefix):
            return provider
    msg = f"Cannot infer a provider for model {model_name!r}; pass --provider."
    raise ValueError(msg)


def _resolve_provider(
    provider: ModelProvider | None, model_name: str | None
) -> ModelProvider:
    if provider is not None:
        return provider
    if model_name is not None:
        return _infer_provider(model_name)
    return "anthropic"


def _create_model(
    provider: ModelProvider | None = None, model_name: str | None = None
) -> BaseChatModel:
    """Create a chat model for the requested provider.

    Args:
        provider: Model provider to use. Inferred from `model_name` when omitted.
        model_name: Provider-specific model name. Uses the provider default when omitted.

    Returns:
        A configured LangChain chat model.

    Raises:
        ValueError: If `provider` is unsupported.
    """
    resolved_provider = _resolve_provider(provider, model_name)
    try:
        resolved_model = model_name or DEFAULT_MODELS[resolved_provider]
        factory = MODEL_FACTORIES[resolved_provider]
    except KeyError:
        msg = f"Unsupported model provider: {resolved_provider}"
        raise ValueError(msg)
    return factory(resolved_model)


def create_sql_deep_agent(
    *,
    provider: ModelProvider | None = None,
    model_name: str | None = None,
) -> CompiledStateGraph:
    """Create a text-to-SQL Deep Agent.

    Args:
        provider: Model provider to use. Inferred from `model_name` when omitted.
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
  python agent.py --provider kimi "How many customers are from Canada?"
  python agent.py --model gemini-3.7-flash "Which employee generated the most revenue?"
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
        help="Model provider (default: inferred from --model, otherwise anthropic)",
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
    provider = cast(ModelProvider | None, args.provider)
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
