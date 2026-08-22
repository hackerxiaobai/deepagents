from unittest import TestCase
from unittest.mock import MagicMock, patch

from agent import (
    DEFAULT_MODELS,
    _build_trace_tags,
    _create_model,
    _infer_provider,
    create_sql_deep_agent,
    main,
)


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

        chat_openai.assert_called_once_with(
            model=DEFAULT_MODELS["openai"], use_responses_api=True
        )
        self.assertIs(model, chat_openai.return_value)

    @patch("agent.ChatOpenAI")
    def test_create_model_accepts_openai_override(self, chat_openai: MagicMock) -> None:
        _create_model("openai", "gpt-5.6-sol")

        chat_openai.assert_called_once_with(model="gpt-5.6-sol", use_responses_api=True)

    @patch.dict("agent.os.environ", {"MOONSHOT_API_KEY": "test-key"})
    @patch("agent.ChatOpenAI")
    def test_create_model_configures_kimi_endpoint(
        self, chat_openai: MagicMock
    ) -> None:
        _create_model("kimi")

        kwargs = chat_openai.call_args.kwargs
        self.assertEqual(kwargs["model"], DEFAULT_MODELS["kimi"])
        self.assertEqual(kwargs["api_key"].get_secret_value(), "test-key")
        self.assertEqual(
            kwargs["base_url"],
            "https://api.moonshot.ai/v1",
        )

    @patch.dict("agent.os.environ", {"ZAI_API_KEY": "test-key"})
    @patch("agent.ChatOpenAI")
    def test_create_model_configures_glm_endpoint(self, chat_openai: MagicMock) -> None:
        _create_model("glm")

        kwargs = chat_openai.call_args.kwargs
        self.assertEqual(kwargs["model"], DEFAULT_MODELS["glm"])
        self.assertEqual(kwargs["api_key"].get_secret_value(), "test-key")
        self.assertEqual(
            kwargs["base_url"],
            "https://api.z.ai/api/paas/v4/",
        )

    @patch("agent.ChatDeepSeek")
    def test_create_model_uses_deepseek_integration(
        self, chat_deepseek: MagicMock
    ) -> None:
        _create_model("deepseek")

        chat_deepseek.assert_called_once_with(model=DEFAULT_MODELS["deepseek"])

    @patch("agent.ChatGoogleGenerativeAI")
    def test_create_model_uses_gemini_integration(self, chat_gemini: MagicMock) -> None:
        _create_model("gemini")

        chat_gemini.assert_called_once_with(model=DEFAULT_MODELS["gemini"])

    @patch("agent.ChatQwen")
    def test_create_model_uses_qwen_integration(self, chat_qwen: MagicMock) -> None:
        _create_model("qwen")

        chat_qwen.assert_called_once_with(model=DEFAULT_MODELS["qwen"])

    @patch("agent.ChatXAI")
    def test_create_model_uses_grok_integration(self, chat_xai: MagicMock) -> None:
        _create_model("grok")

        chat_xai.assert_called_once_with(model=DEFAULT_MODELS["grok"])

    def test_create_model_requires_kimi_key(self) -> None:
        with (
            patch.dict("agent.os.environ", {}, clear=True),
            self.assertRaisesRegex(ValueError, "Set MOONSHOT_API_KEY"),
        ):
            _create_model("kimi")

    def test_infer_provider_from_model_name(self) -> None:
        cases = {
            "claude-haiku-4-5-20251001": "anthropic",
            "gpt-5.6-sol": "openai",
            "kimi-k3": "kimi",
            "glm-5.1": "glm",
            "deepseek-v4-pro": "deepseek",
            "gemini-3.7-flash": "gemini",
            "qwen3.8-max": "qwen",
            "grok-4.6": "grok",
        }
        for model_name, provider in cases.items():
            with self.subTest(model_name=model_name):
                self.assertEqual(_infer_provider(model_name), provider)

    def test_build_trace_tags_uses_provider_default_model(self) -> None:
        tags = _build_trace_tags("kimi", None)

        self.assertEqual(
            tags,
            [
                "app:text-to-sql",
                "provider:kimi",
                f"model:{DEFAULT_MODELS['kimi']}",
            ],
        )

    def test_build_trace_tags_infers_provider_from_model(self) -> None:
        tags = _build_trace_tags(None, "glm-5.3")

        self.assertEqual(
            tags,
            ["app:text-to-sql", "provider:glm", "model:glm-5.3"],
        )

    @patch("agent.ChatGoogleGenerativeAI")
    def test_create_model_infers_provider(self, chat_gemini: MagicMock) -> None:
        _create_model(model_name="gemini-3.7-flash")

        chat_gemini.assert_called_once_with(model="gemini-3.7-flash")

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

    @patch("agent.console")
    @patch("agent.create_sql_deep_agent")
    @patch(
        "agent.sys.argv",
        ["agent.py", "--provider", "kimi", "How many customers are there?"],
    )
    def test_main_adds_model_trace_tags(
        self, create_agent: MagicMock, _console: MagicMock
    ) -> None:
        create_agent.return_value.invoke.return_value = {
            "messages": [MagicMock(content="There are 59 customers.")]
        }

        main()

        create_agent.return_value.invoke.assert_called_once_with(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "How many customers are there?",
                    }
                ]
            },
            config={
                "tags": [
                    "app:text-to-sql",
                    "provider:kimi",
                    f"model:{DEFAULT_MODELS['kimi']}",
                ]
            },
        )
