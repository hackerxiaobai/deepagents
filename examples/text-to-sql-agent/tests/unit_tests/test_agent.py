from unittest import TestCase
from unittest.mock import MagicMock, patch

from agent import DEFAULT_MODELS, _create_model, create_sql_deep_agent


class ModelValidationTests(TestCase):
    @patch("agent.ChatAnthropic")
    def test_create_model_uses_anthropic_default(
        self, chat_anthropic: MagicMock
    ) -> None:
        model = _create_model("anthropic")

        chat_anthropic.assert_called_once_with(
            model_name=DEFAULT_MODELS["anthropic"], temperature=0
        )
        self.assertIs(model, chat_anthropic.return_value)

    @patch("agent.ChatOpenAI")
    def test_create_model_uses_openai_default(self, chat_openai: MagicMock) -> None:
        model = _create_model("openai")

        chat_openai.assert_called_once_with(model=DEFAULT_MODELS["openai"])
        self.assertIs(model, chat_openai.return_value)

    @patch("agent.ChatOpenAI")
    def test_create_model_accepts_openai_override(self, chat_openai: MagicMock) -> None:
        _create_model("openai", "gpt-5.6-sol")

        chat_openai.assert_called_once_with(model="gpt-5.6-sol")

    def test_create_model_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported model provider: unknown"):
            _create_model("unknown")  # type: ignore[arg-type]

    @patch("agent.create_deep_agent")
    @patch("agent.SQLDatabaseToolkit")
    @patch("agent.SQLDatabase.from_uri")
    @patch("agent._create_model")
    def test_create_agent_uses_selected_model_for_toolkit_and_agent(
        self,
        create_model: MagicMock,
        from_uri: MagicMock,
        toolkit_class: MagicMock,
        create_deep_agent: MagicMock,
    ) -> None:
        graph = create_sql_deep_agent(provider="openai", model_name="gpt-5.6-sol")

        create_model.assert_called_once_with("openai", "gpt-5.6-sol")
        toolkit_class.assert_called_once_with(
            db=from_uri.return_value, llm=create_model.return_value
        )
        self.assertIs(
            create_deep_agent.call_args.kwargs["model"], create_model.return_value
        )
        self.assertIs(graph, create_deep_agent.return_value)
