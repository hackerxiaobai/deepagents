from unittest import TestCase
from unittest.mock import MagicMock, patch

from agent_subagent_tool_exclusion import (
    EXCLUDED_TOOLS,
    HARNESS_PROVIDERS,
    _register_restricted_harness_profiles,
    create_restricted_sql_subagent_graph,
)


class ToolExclusionTests(TestCase):
    def test_excluded_tools_keep_required_read_and_planning_tools(self) -> None:
        self.assertEqual(
            EXCLUDED_TOOLS,
            frozenset(
                {
                    "ls",
                    "write_file",
                    "edit_file",
                    "glob",
                    "grep",
                    "execute",
                }
            ),
        )
        self.assertNotIn("read_file", EXCLUDED_TOOLS)
        self.assertNotIn("write_todos", EXCLUDED_TOOLS)
        self.assertNotIn("task", EXCLUDED_TOOLS)

    @patch("agent_subagent_tool_exclusion.register_harness_profile")
    def test_register_profiles_disables_default_subagent_and_tools(
        self,
        register_profile: MagicMock,
    ) -> None:
        _register_restricted_harness_profiles()

        self.assertEqual(register_profile.call_count, len(HARNESS_PROVIDERS))
        for call in register_profile.call_args_list:
            profile = call.args[1]
            self.assertFalse(profile.general_purpose_subagent.enabled)
            self.assertEqual(profile.excluded_tools, EXCLUDED_TOOLS)

    @patch("agent_subagent_tool_exclusion.create_deep_agent")
    @patch("agent_subagent_tool_exclusion._register_restricted_harness_profiles")
    @patch("agent_subagent_tool_exclusion._build_sql_subagents")
    @patch("agent_subagent_tool_exclusion.SQLDatabaseToolkit")
    @patch("agent_subagent_tool_exclusion.SQLDatabase.from_uri")
    @patch("agent_subagent_tool_exclusion._create_model")
    def test_create_graph_registers_restrictions_before_agent_creation(
        self,
        create_model: MagicMock,
        from_uri: MagicMock,
        toolkit_class: MagicMock,
        build_subagents: MagicMock,
        register_profiles: MagicMock,
        create_deep_agent: MagicMock,
    ) -> None:
        sql_tools = toolkit_class.return_value.get_tools.return_value

        graph = create_restricted_sql_subagent_graph(
            provider="openai",
            model_name="test-model",
        )

        register_profiles.assert_called_once_with()
        build_subagents.assert_called_once_with(sql_tools)
        kwargs = create_deep_agent.call_args.kwargs
        self.assertEqual(kwargs["tools"], [])
        self.assertIs(kwargs["model"], create_model.return_value)
        self.assertIs(graph, create_deep_agent.return_value)
        toolkit_class.assert_called_once_with(
            db=from_uri.return_value,
            llm=create_model.return_value,
        )
