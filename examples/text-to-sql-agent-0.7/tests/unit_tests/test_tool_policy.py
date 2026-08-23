from pathlib import Path
from typing import cast
from unittest import TestCase
from unittest.mock import MagicMock, patch

from deepagents import FilesystemMiddleware
from deepagents.backends import StateBackend

from tool_policy import (
    HARNESS_PROVIDERS,
    READ_ONLY_FILESYSTEM_TOOLS,
    create_read_only_filesystem_middleware,
    register_read_only_sql_harness_profiles,
)


class ToolPolicyTests(TestCase):
    def test_prompt_sources_do_not_reference_removed_write_todos_tool(self) -> None:
        project_root = Path(__file__).parents[2]
        prompt_sources = (
            project_root / "AGENTS.md",
            project_root / "skills/query-writing/SKILL.md",
            project_root / "skills/schema-exploration/SKILL.md",
        )

        for prompt_source in prompt_sources:
            with self.subTest(prompt_source=prompt_source):
                self.assertNotIn("write_todos", prompt_source.read_text())

    def test_filesystem_allowlist_exposes_only_read_file(self) -> None:
        middleware = create_read_only_filesystem_middleware(StateBackend())
        filesystem_middleware = cast("FilesystemMiddleware", middleware)

        self.assertEqual(READ_ONLY_FILESYSTEM_TOOLS, ["read_file"])
        self.assertEqual(
            [tool.name for tool in filesystem_middleware.tools],
            ["read_file"],
        )

    @patch("tool_policy.register_harness_profile")
    def test_profile_disables_general_purpose_subagent(
        self,
        register_profile: MagicMock,
    ) -> None:
        register_read_only_sql_harness_profiles()

        self.assertEqual(register_profile.call_count, len(HARNESS_PROVIDERS))
        for call in register_profile.call_args_list:
            profile = call.args[1]
            self.assertFalse(profile.general_purpose_subagent.enabled)
