"""Text-to-SQL Deep Agent composed from specialized SQL subagents."""

import argparse
import os
import sys
from collections.abc import Sequence
from typing import cast

import ipdb
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.panel import Panel

from agent import DEFAULT_MODELS, ModelProvider, _build_trace_tags, _create_model
from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    SubAgent,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import FilesystemBackend

console = Console()

HARNESS_PROVIDERS = (
    "anthropic",
    "openai",
    "deepseek",
    "google_genai",
    "xai",
)

ORCHESTRATOR_PROMPT = """You coordinate read-only SQLite analysis.
Delegate schema discovery to `schema-explorer` and SQL query work to
`sql-query-writer`. Synthesize their results into a concise final answer.
Never invent schema details or execute database writes."""


def _disable_general_purpose_subagent() -> None:
    """Disable the automatically added general-purpose subagent."""
    profile = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
    )
    for provider in HARNESS_PROVIDERS:
        register_harness_profile(provider, profile)


def _select_tools(tools: Sequence[BaseTool], names: Sequence[str]) -> list[BaseTool]:
    """Select required tools by name and fail clearly when one is unavailable."""
    tools_by_name = {tool.name: tool for tool in tools}
    missing = [name for name in names if name not in tools_by_name]
    if missing:
        msg = f"SQL toolkit is missing required tools: {', '.join(missing)}"
        raise ValueError(msg)
    return [tools_by_name[name] for name in names]


def _build_sql_subagents(sql_tools: Sequence[BaseTool]) -> list[SubAgent]:
    """Build specialized schema exploration and query execution subagents."""
    schema_agent: SubAgent = {
        "name": "schema-explorer",
        "description": (
            "Explores SQLite tables, columns, keys, samples, and relationships. "
            "Use when the relevant schema is unknown or the user asks about structure."
        ),
        "system_prompt": (
            "Inspect the database schema using only the available schema tools. "
            "Report relevant tables, columns, keys, and relationships. "
            "Do not execute business queries or modify the database."
        ),
        "tools": _select_tools(
            sql_tools,
            ("sql_db_list_tables", "sql_db_schema"),
        ),
        "skills": ["./skills/"],
    }
    query_agent: SubAgent = {
        "name": "sql-query-writer",
        "description": (
            "Writes, validates, and executes read-only SQLite queries. "
            "Use for questions that require retrieving or aggregating database data."
        ),
        "system_prompt": (
            "Answer database questions with read-only SQLite SELECT statements. "
            "Inspect schemas before relying on columns, validate SQL before execution, "
            "and limit results to 5 rows unless the user requests otherwise. Never use "
            "INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, or CREATE."
        ),
        "tools": _select_tools(
            sql_tools,
            (
                "sql_db_list_tables",
                "sql_db_schema",
                "sql_db_query_checker",
                "sql_db_query",
            ),
        ),
        "skills": ["./skills/"],
    }
    return [schema_agent, query_agent]


def create_sql_subagent_graph(
    *,
    provider: ModelProvider | None = None,
    model_name: str | None = None,
) -> CompiledStateGraph:
    """Create a text-to-SQL graph backed by specialized subagents."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "chinook.db")
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
    model = _create_model(provider, model_name)
    sql_tools = SQLDatabaseToolkit(db=db, llm=model).get_tools()

    _disable_general_purpose_subagent()
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
        description="Text-to-SQL Deep Agent with specialized subagents",
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
    """Run the specialized text-to-SQL Deep Agent CLI."""
    args = _build_parser().parse_args()
    provider = cast(ModelProvider | None, args.provider)
    console.print(
        Panel(f"[bold cyan]Question:[/bold cyan] {args.question}", border_style="cyan")
    )
    console.print("\n[dim]Creating SQL subagents and processing query...[/dim]\n")

    try:
        graph = create_sql_subagent_graph(
            provider=provider,
            model_name=args.model_name,
        )
        invoke_config: RunnableConfig = {
            "tags": [
                *_build_trace_tags(provider, args.model_name),
                "architecture:subagents",
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
