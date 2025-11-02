import sys
import types
import unittest
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_reflex_stub() -> None:
    """Provide a lightweight stub for the reflex package if it's missing."""
    if "reflex" in sys.modules:
        return

    class _FieldAnnotation:
        def __class_getitem__(cls, _item: Any) -> "_FieldAnnotation":
            return cls

    class _FieldDescriptor:
        def __init__(
            self, default: Any = None, default_factory: Callable[[], Any] | None = None
        ):
            self.default = default
            self.default_factory = default_factory
            self.name: str | None = None

        def __set_name__(self, _owner: type, name: str) -> None:
            self.name = name

        def _initial_value(self) -> Any:
            if self.default_factory is not None:
                return self.default_factory()
            return self.default

        def __get__(self, instance: Any, owner: type) -> Any:
            if instance is None:
                return self
            assert self.name is not None
            if self.name not in instance.__dict__:
                instance.__dict__[self.name] = self._initial_value()
            return instance.__dict__[self.name]

        def __set__(self, instance: Any, value: Any) -> None:
            assert self.name is not None
            instance.__dict__[self.name] = value

    def field(
        default: Any = None, *, default_factory: Callable[[], Any] | None = None
    ):
        return _FieldDescriptor(default=default, default_factory=default_factory)

    def event(fn: Callable):
        return fn

    def var(fn: Callable):
        return property(fn)

    reflex = types.ModuleType("reflex")
    reflex.State = type("State", (), {})
    reflex.Field = _FieldAnnotation
    reflex.field = field
    reflex.event = event
    reflex.var = var

    # Minimal style submodule to satisfy imports elsewhere if needed.
    style = types.ModuleType("reflex.style")
    style.color_mode = "system"

    sys.modules["reflex"] = reflex
    sys.modules["reflex.style"] = style


def _ensure_pydantic_ai_stub() -> None:
    """Provide a minimal Agent stub when pydantic_ai is unavailable."""
    if "pydantic_ai" in sys.modules:
        return

    class Agent:
        def __init__(self, model: str):
            self.model = model

        async def run(self, _prompt: str):
            return types.SimpleNamespace(output="")

    module = types.ModuleType("pydantic_ai")
    module.Agent = Agent
    sys.modules["pydantic_ai"] = module


_ensure_reflex_stub()
_ensure_pydantic_ai_stub()

from app.state import SettingsState, State


class StateTests(unittest.TestCase):
    def setUp(self) -> None:
        """Create a fresh State instance for each test."""
        self.state = State()

    def test_create_new_session_initializes_state(self) -> None:
        """create_new_session seeds the first chat and marks it active."""
        self.assertEqual(self.state.chat_sessions, [])
        self.assertEqual(self.state.current_session_id, "")

        self.state.create_new_session()

        self.assertEqual(len(self.state.chat_sessions), 1)
        session = self.state.chat_sessions[0]
        self.assertEqual(self.state.current_session_id, session.id)
        self.assertEqual(session.title, "New Chat")
        self.assertEqual(session.messages, [])

    def test_switch_session_invalid_id_falls_back_to_first(self) -> None:
        """switch_session ignores stale ids and falls back gracefully."""
        self.state.create_new_session()
        first_id = self.state.current_session_id
        self.state.create_new_session()

        self.state.switch_session("non-existent-id")

        self.assertEqual(self.state.current_session_id, first_id)

    def test_delete_session_moves_to_first_remaining(self) -> None:
        """delete_session keeps state consistent when other chats remain."""
        self.state.create_new_session()
        first_id = self.state.current_session_id
        self.state.create_new_session()

        self.state.delete_session(self.state.current_session_id)

        self.assertEqual(len(self.state.chat_sessions), 1)
        self.assertEqual(self.state.current_session_id, first_id)

    def test_delete_session_creates_new_when_none_left(self) -> None:
        """delete_session bootstraps a replacement when all chats are removed."""
        self.state.create_new_session()
        original_id = self.state.current_session_id

        self.state.delete_session(original_id)

        self.assertEqual(len(self.state.chat_sessions), 1)
        self.assertNotEqual(self.state.current_session_id, original_id)

    def test_update_session_title_truncates_long_question(self) -> None:
        """update_session_title shortens long prompts for sidebar display."""
        self.state.create_new_session()
        long_question = "What strategies can improve developer productivity in large distributed teams?"  # noqa: E501

        self.state.update_session_title(long_question)

        expected_title = long_question[:50] + "..."
        self.assertEqual(self.state.chat_sessions[0].title, expected_title)

    def test_chat_history_returns_current_session_messages(self) -> None:
        """chat_history returns the active session's shared message list."""
        self.state.create_new_session()
        session = self.state.chat_sessions[0]
        session.messages.append(("Q1", "A1"))
        session.messages.append(("Q2", "A2"))

        history = self.state.chat_history

        self.assertIs(history, session.messages)
        self.assertEqual(len(history), 2)

    def test_chat_history_is_empty_when_no_session(self) -> None:
        """chat_history is empty before any sessions are created."""
        self.assertEqual(self.state.chat_history, [])

    def test_sidebar_classes_reflects_sidebar_state(self) -> None:
        """sidebar_classes toggles the translation utility via sidebar_open."""
        open_classes = self.state.sidebar_classes
        self.assertIn("translate-x-0", open_classes)

        self.state.toggle_sidebar()

        closed_classes = self.state.sidebar_classes
        self.assertIn("-translate-x-full", closed_classes)

    def test_clear_chat_creates_new_session_and_resets_processing(self) -> None:
        """clear_chat ends processing and starts a fresh conversation."""
        self.state.create_new_session()
        self.state.processing = True
        previous_id = self.state.current_session_id

        self.state.clear_chat()

        self.assertFalse(self.state.processing)
        self.assertGreaterEqual(len(self.state.chat_sessions), 2)
        self.assertNotEqual(self.state.current_session_id, previous_id)
        current_session = self.state.chat_sessions[-1]
        self.assertEqual(current_session.id, self.state.current_session_id)
        self.assertEqual(current_session.messages, [])


class SettingsStateTests(unittest.TestCase):
    def test_set_font_family_updates_state(self) -> None:
        """set_font_family persists the user's selected typography."""
        settings_state = SettingsState()
        self.assertEqual(settings_state.font_family, "Poppins")

        settings_state.set_font_family("Inter")

        self.assertEqual(settings_state.font_family, "Inter")


if __name__ == "__main__":
    unittest.main()
