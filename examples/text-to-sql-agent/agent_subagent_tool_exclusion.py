"""Text-to-SQL subagent example with unused built-in tools excluded."""

import argparse
import os
import sys
from typing import cast

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.panel import Panel

from agent import DEFAULT_MODELS, ModelProvider, _build_trace_tags, _create_model
from agent_subagent import HARNESS_PROVIDERS, ORCHESTRATOR_PROMPT, _build_sql_subagents
from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import FilesystemBackend

console = Console()

EXCLUDED_TOOLS = frozenset(
    {
        "ls",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "execute",
    }
)


def _register_restricted_harness_profiles() -> None:
    """Disable the default subagent and hide unused built-in tools."""
    profile = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=EXCLUDED_TOOLS,
    )
    for provider in HARNESS_PROVIDERS:
        register_harness_profile(provider, profile)


def create_restricted_sql_subagent_graph(
    *,
    provider: ModelProvider | None = None,
    model_name: str | None = None,
) -> CompiledStateGraph:
    """Create the SQL subagent graph without unnecessary built-in tools."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "chinook.db")
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
    model = _create_model(provider, model_name)
    sql_tools = SQLDatabaseToolkit(db=db, llm=model).get_tools()

    _register_restricted_harness_profiles()
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt=ORCHESTRATOR_PROMPT,
        memory=["./AGENTS.md"],
        skills=["./skills/"],
        subagents=_build_sql_subagents(sql_tools),
        backend=FilesystemBackend(root_dir=base_dir, virtual_mode=True),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL subagents with unused built-in tools excluded",
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
    return parser


def main() -> None:
    """Run the tool-restricted text-to-SQL subagent CLI."""
    args = _build_parser().parse_args()
    provider = cast(ModelProvider | None, args.provider)
    console.print(
        Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan")
    )
    console.print("\n[dim]Creating tool-restricted SQL subagents...[/dim]\n")

    try:
        graph = create_restricted_sql_subagent_graph(
            provider=provider,
            model_name=args.model_name,
        )
        invoke_config: RunnableConfig = {
            "tags": [
                *_build_trace_tags(provider, args.model_name),
                "architecture:subagents",
                "tools:restricted",
            ]
        }
        result = graph.invoke(
            {"messages": [{"role": "user", "content": args.question}]},
            config=invoke_config,
        )
        final_message = result["messages"][-1]
        answer = getattr(final_message, "content", str(final_message))
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
