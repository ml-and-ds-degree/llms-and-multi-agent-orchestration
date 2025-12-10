"""Unit tests for application state management."""

import sys
import types
from pathlib import Path
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_reflex_stub() -> None:
    """Provide a lightweight stub for the reflex package if it's missing."""
    if "reflex" in sys.modules:
        return

    class _FieldAnnotation:
        def __class_getitem__(cls, _item: Any):
            return cls

    class _FieldDescriptor:
        def __init__(
            self,
            default: Any = None,
            default_factory: Optional[Callable[[], Any]] = None,
        ):
            self.default = default
            self.default_factory = default_factory
            self.name: Optional[str] = None

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
        default: Any = None, *, default_factory: Optional[Callable[[], Any]] = None
    ):
        return _FieldDescriptor(default=default, default_factory=default_factory)

    def event(fn: Optional[Callable] = None, *, background: bool = False):
        """Stub for rx.event decorator that accepts background parameter."""
        def decorator(func: Callable) -> Callable:
            return func
        
        # Handle both @rx.event and @rx.event(background=True)
        if fn is None:
            return decorator
        return decorator(fn)

    def var(fn: Callable):
        return property(fn)

    class State:
        """Base State class with async context manager support."""
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    reflex = types.ModuleType("reflex")
    setattr(reflex, "State", State)
    setattr(reflex, "Field", _FieldAnnotation)
    setattr(reflex, "field", field)
    setattr(reflex, "event", event)
    setattr(reflex, "var", var)

    style = types.ModuleType("reflex.style")
    setattr(style, "color_mode", "system")

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
    setattr(module, "Agent", Agent)
    sys.modules["pydantic_ai"] = module


_ensure_reflex_stub()
_ensure_pydantic_ai_stub()

from app.state import ChatSession, SettingsState, State  # noqa: E402

# ==================== FIXTURES ====================


@pytest.fixture
def state():
    """Create a fresh State instance for each test."""
    return State()


@pytest.fixture
def settings_state():
    """Create a fresh SettingsState instance for each test."""
    return SettingsState()


@pytest.fixture
def state_with_session(state):
    """Create a State with one active session."""
    state.create_new_session()
    return state


@pytest.fixture
def state_with_messages(state_with_session):
    """Create a State with a session containing messages."""
    session = state_with_session.chat_sessions[0]
    session.messages.append(("What is Python?", "Python is a programming language."))
    session.messages.append(("What is Reflex?", "Reflex is a web framework."))
    return state_with_session


@pytest.fixture
def mock_agent():
    """Mock the pydantic_ai Agent for testing async methods."""
    mock = MagicMock()
    mock_result = AsyncMock()
    mock_result.stream_text = AsyncMock(
        return_value=iter(["Hello", "Hello world", "Hello world!"])
    )
    mock_result.__aenter__ = AsyncMock(return_value=mock_result)
    mock_result.__aexit__ = AsyncMock(return_value=None)
    mock.run_stream = MagicMock(return_value=mock_result)
    return mock


# ==================== SETTINGS STATE TESTS ====================


class TestSettingsState:
    """Test suite for SettingsState."""

    def test_default_font_family(self, settings_state):
        """Default font family should be Poppins."""
        assert settings_state.font_family == "Poppins"

    def test_set_font_family(self, settings_state):
        """set_font_family should update the font_family attribute."""
        settings_state.set_font_family("Inter")
        assert settings_state.font_family == "Inter"

    def test_set_font_family_multiple_times(self, settings_state):
        """Font family should update correctly on multiple calls."""
        settings_state.set_font_family("Roboto")
        assert settings_state.font_family == "Roboto"

        settings_state.set_font_family("Arial")
        assert settings_state.font_family == "Arial"


# ==================== CHAT SESSION TESTS ====================


class TestChatSession:
    """Test suite for ChatSession dataclass."""

    def test_create_chat_session(self):
        """ChatSession should initialize with correct attributes."""
        session = ChatSession(
            id="123", title="Test Chat", created_at="Jan 1, 2024", messages=[]
        )
        assert session.id == "123"
        assert session.title == "Test Chat"
        assert session.created_at == "Jan 1, 2024"
        assert session.messages == []

    def test_chat_session_default_messages(self):
        """ChatSession should default to empty messages list."""
        session = ChatSession(id="456", title="New Chat", created_at="Jan 2, 2024")
        assert session.messages == []


# ==================== STATE - SESSION MANAGEMENT TESTS ====================


class TestStateSessionManagement:
    """Test suite for session creation, switching, and deletion."""

    def test_initial_state(self, state):
        """State should initialize with no sessions."""
        assert state.chat_sessions == []
        assert state.current_session_id == ""
        assert state.sidebar_open is True
        assert state.processing is False

    def test_create_new_session(self, state):
        """create_new_session should create and activate a new session."""
        state.create_new_session()

        assert len(state.chat_sessions) == 1
        session = state.chat_sessions[0]
        assert state.current_session_id == session.id
        assert session.title == "New Chat"
        assert session.messages == []
        assert session.id != ""

    def test_create_multiple_sessions(self, state):
        """Should be able to create multiple sessions."""
        state.create_new_session()
        first_id = state.current_session_id

        state.create_new_session()
        second_id = state.current_session_id

        assert len(state.chat_sessions) == 2
        assert first_id != second_id
        assert state.current_session_id == second_id

    def test_switch_session_valid_id(self, state):
        """switch_session should change to existing session."""
        state.create_new_session()
        first_id = state.current_session_id
        state.create_new_session()
        second_id = state.current_session_id

        state.switch_session(first_id)
        assert state.current_session_id == first_id

    def test_switch_session_invalid_id_falls_back(self, state):
        """switch_session with invalid ID should fall back to first session."""
        state.create_new_session()
        first_id = state.current_session_id
        state.create_new_session()

        state.switch_session("non-existent-id")
        assert state.current_session_id == first_id

    def test_switch_session_invalid_id_no_sessions(self, state):
        """switch_session with invalid ID and no sessions should clear current_session_id."""
        state.switch_session("non-existent-id")
        assert state.current_session_id == ""

    def test_delete_session_switches_to_first_remaining(self, state):
        """delete_session should switch to first remaining session."""
        state.create_new_session()
        first_id = state.current_session_id
        state.create_new_session()
        second_id = state.current_session_id

        state.delete_session(second_id)

        assert len(state.chat_sessions) == 1
        assert state.current_session_id == first_id

    def test_delete_session_creates_new_when_none_left(self, state):
        """delete_session should create new session when deleting last one."""
        state.create_new_session()
        original_id = state.current_session_id

        state.delete_session(original_id)

        assert len(state.chat_sessions) == 1
        assert state.current_session_id != original_id
        assert state.current_session_id != ""

    def test_delete_non_current_session(self, state):
        """Deleting non-current session should preserve current session."""
        state.create_new_session()
        first_id = state.current_session_id
        state.create_new_session()
        second_id = state.current_session_id

        state.delete_session(first_id)

        assert len(state.chat_sessions) == 1
        assert state.current_session_id == second_id


# ==================== STATE - COMPUTED PROPERTIES TESTS ====================


class TestStateComputedProperties:
    """Test suite for rx.var computed properties."""

    def test_current_session_none_when_no_sessions(self, state):
        """current_session should return None when no sessions exist."""
        assert state.current_session is None

    def test_current_session_returns_active_session(self, state_with_session):
        """current_session should return the active session."""
        session = state_with_session.current_session
        assert session is not None
        assert session.id == state_with_session.current_session_id

    def test_chat_history_empty_when_no_session(self, state):
        """chat_history should be empty when no session exists."""
        assert state.chat_history == []

    def test_chat_history_returns_current_messages(self, state_with_messages):
        """chat_history should return messages from current session."""
        history = state_with_messages.chat_history
        assert len(history) == 2
        assert history[0] == ("What is Python?", "Python is a programming language.")
        assert history[1] == ("What is Reflex?", "Reflex is a web framework.")

    def test_sidebar_classes_when_open(self, state):
        """sidebar_classes should include translate-x-0 when open."""
        state.sidebar_open = True
        classes = state.sidebar_classes
        assert "translate-x-0" in classes
        assert "bg-slate-2" in classes
        assert "transition-transform" in classes

    def test_sidebar_classes_when_closed(self, state):
        """sidebar_classes should include -translate-x-full when closed."""
        state.sidebar_open = False
        classes = state.sidebar_classes
        assert "-translate-x-full" in classes


# ==================== STATE - UI INTERACTIONS TESTS ====================


class TestStateUIInteractions:
    """Test suite for UI interaction event handlers."""

    def test_set_question(self, state):
        """set_question should update the question field."""
        state.set_question("What is AI?")
        assert state.question == "What is AI?"

    def test_toggle_sidebar_from_open_to_closed(self, state):
        """toggle_sidebar should close an open sidebar."""
        state.sidebar_open = True
        state.toggle_sidebar()
        assert state.sidebar_open is False

    def test_toggle_sidebar_from_closed_to_open(self, state):
        """toggle_sidebar should open a closed sidebar."""
        state.sidebar_open = False
        state.toggle_sidebar()
        assert state.sidebar_open is True

    def test_clear_chat_creates_new_session(self, state_with_messages):
        """clear_chat should create a new session."""
        original_session_count = len(state_with_messages.chat_sessions)
        original_id = state_with_messages.current_session_id

        state_with_messages.clear_chat()

        assert len(state_with_messages.chat_sessions) == original_session_count + 1
        assert state_with_messages.current_session_id != original_id

    def test_clear_chat_resets_processing(self, state_with_session):
        """clear_chat should reset processing flag."""
        state_with_session.processing = True
        state_with_session.clear_chat()
        assert state_with_session.processing is False


# ==================== STATE - SESSION TITLE TESTS ====================


class TestStateSessionTitle:
    """Test suite for session title management."""

    def test_update_session_title_short_question(self, state_with_session):
        """update_session_title should use full text for short questions."""
        short_question = "What is Python?"
        state_with_session.update_session_title(short_question)

        session = state_with_session.chat_sessions[0]
        assert session.title == short_question

    def test_update_session_title_long_question(self, state_with_session):
        """update_session_title should truncate long questions."""
        long_question = "What strategies can improve developer productivity in large distributed teams?"
        state_with_session.update_session_title(long_question)

        session = state_with_session.chat_sessions[0]
        expected_title = long_question[:50] + "..."
        assert session.title == expected_title
        assert len(session.title) == 53  # 50 chars + "..."

    def test_update_session_title_exactly_50_chars(self, state_with_session):
        """update_session_title should not truncate exactly 50 char questions."""
        question_50 = "a" * 50
        state_with_session.update_session_title(question_50)

        session = state_with_session.chat_sessions[0]
        assert session.title == question_50
        assert "..." not in session.title


# ==================== STATE - ASYNC ANSWER TESTS ====================


class TestStateAnswerMethod:
    """Test suite for the async answer method."""

    @pytest.mark.asyncio
    async def test_answer_empty_question_returns_early(self, state):
        """answer should return early if question is empty."""
        state.question = ""
        await state.answer()
        assert len(state.chat_sessions) == 0

    @pytest.mark.asyncio
    async def test_answer_whitespace_question_returns_early(self, state):
        """answer should return early if question is only whitespace."""
        state.question = "   "
        await state.answer()
        assert len(state.chat_sessions) == 0

    @pytest.mark.asyncio
    async def test_answer_creates_session_if_none_exists(self, state, mock_agent):
        """answer should create a new session if none exists."""
        state.question = "Hello"

        with patch("app.state.Agent", return_value=mock_agent):
            await state.answer()

        assert len(state.chat_sessions) >= 1
        assert state.current_session_id != ""

    @pytest.mark.asyncio
    async def test_handle_key_down_enter_triggers_answer(self, state):
        """handle_key_down should return answer event on Enter key."""
        state.question = "Test"
        state.processing = False

        result = await state.handle_key_down("Enter")
        assert result == State.answer

    @pytest.mark.asyncio
    async def test_handle_key_down_other_key_no_action(self, state):
        """handle_key_down should not trigger on other keys."""
        state.question = "Test"
        result = await state.handle_key_down("a")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_key_down_enter_while_processing(self, state):
        """handle_key_down should not trigger while processing."""
        state.question = "Test"
        state.processing = True

        result = await state.handle_key_down("Enter")
        assert result is None
