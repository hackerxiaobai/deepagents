"""Shared read-only built-in tool policy for the text-to-SQL agents."""

from typing import cast

from deepagents import (
    FilesystemMiddleware,
    FsToolName,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    register_harness_profile,
)
from deepagents.backends import BackendProtocol
from langchain.agents.middleware.types import AgentMiddleware

READ_ONLY_FILESYSTEM_TOOLS: list[FsToolName] = ["read_file"]
HARNESS_PROVIDERS = (
    "anthropic",
    "openai",
    "deepseek",
    "google_genai",
    "xai",
)


def create_read_only_filesystem_middleware(
    backend: BackendProtocol,
) -> AgentMiddleware:
    """Expose only the file reader from Deep Agents' filesystem tool set."""
    return cast(
        "AgentMiddleware",
        FilesystemMiddleware(
            backend=backend,
            tools=READ_ONLY_FILESYSTEM_TOOLS.copy(),
        ),
    )


def register_read_only_sql_harness_profiles() -> None:
    """Disable the unused automatic general-purpose subagent."""
    profile = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
    )
    for provider in HARNESS_PROVIDERS:
        register_harness_profile(provider, profile)
