from unittest import TestCase
from unittest.mock import MagicMock, patch

from agent_subagent import (
    HARNESS_PROVIDERS,
    _build_sql_subagents,
    _disable_general_purpose_subagent,
    _select_tools,
    create_sql_subagent_graph,
)


def _sql_tools() -> list[MagicMock]:
    names = (
        "sql_db_list_tables",
        "sql_db_schema",
        "sql_db_query_checker",
        "sql_db_query",
    )
    tools = []
    for name in names:
        tool = MagicMock()
        tool.name = name
        tools.append(tool)
    return tools


class SqlSubagentTests(TestCase):
    def test_select_tools_rejects_missing_tool(self) -> None:
        with self.assertRaisesRegex(ValueError, "sql_db_schema"):
            _select_tools([], ("sql_db_schema",))

    def test_build_sql_subagents_assigns_specialized_tools(self) -> None:
        subagents = _build_sql_subagents(_sql_tools())

        self.assertEqual(
            [subagent["name"] for subagent in subagents],
            ["schema-explorer", "sql-query-writer"],
        )
        self.assertEqual(
            [tool.name for tool in subagents[0]["tools"]],
            ["sql_db_list_tables", "sql_db_schema"],
        )
        self.assertEqual(len(subagents[1]["tools"]), 4)

    @patch("agent_subagent.register_harness_profile")
    def test_disable_general_purpose_for_supported_harnesses(
        self, register_profile: MagicMock
    ) -> None:
        _disable_general_purpose_subagent()

        self.assertEqual(register_profile.call_count, len(HARNESS_PROVIDERS))
        for call in register_profile.call_args_list:
            profile = call.args[1]
            self.assertFalse(profile.general_purpose_subagent.enabled)

    @patch("agent_subagent.create_deep_agent")
    @patch("agent_subagent._disable_general_purpose_subagent")
    @patch("agent_subagent.SQLDatabaseToolkit")
    @patch("agent_subagent.SQLDatabase.from_uri")
    @patch("agent_subagent._create_model")
    def test_create_graph_uses_only_custom_sql_subagents(
        self,
        create_model: MagicMock,
        from_uri: MagicMock,
        toolkit_class: MagicMock,
        disable_default: MagicMock,
        create_deep_agent: MagicMock,
    ) -> None:
        toolkit_class.return_value.get_tools.return_value = _sql_tools()

        graph = create_sql_subagent_graph(provider="openai", model_name="test-model")

        disable_default.assert_called_once_with()
        kwargs = create_deep_agent.call_args.kwargs
        self.assertEqual(kwargs["tools"], [])
        self.assertEqual(
            [subagent["name"] for subagent in kwargs["subagents"]],
            ["schema-explorer", "sql-query-writer"],
        )
        self.assertIs(kwargs["model"], create_model.return_value)
        toolkit_class.assert_called_once_with(
            db=from_uri.return_value,
            llm=create_model.return_value,
        )
        self.assertIs(graph, create_deep_agent.return_value)
