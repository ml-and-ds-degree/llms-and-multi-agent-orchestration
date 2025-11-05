"""Reactive application state management for the chatbot experience."""

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import reflex as rx
from pydantic_ai import Agent

os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"


class SettingsState(rx.State):
    """Manage user-configurable interface preferences."""

    # The font family for the app
    font_family: rx.Field[str] = rx.field("Poppins")

    @rx.event
    def set_font_family(self, font_family: str):
        """Update the active font family for the UI.

        Args:
            font_family: Name of the font the interface should use.
        """
        self.font_family = font_family


@dataclass
class ChatSession:
    """Represents a single chat session."""

    id: str
    title: str
    created_at: str
    messages: list[tuple[str, str]] = field(default_factory=list)


class State(rx.State):
    """Track chat flow, session management, and UI state."""

    # The current question being asked.
    question: rx.Field[str] = rx.field("")

    # Whether the app is processing a question.
    processing: rx.Field[bool] = rx.field(False)

    # List of all chat sessions
    chat_sessions: rx.Field[list[ChatSession]] = rx.field(default_factory=list)

    # Current active session ID
    current_session_id: rx.Field[str] = rx.field("")

    # Whether sidebar is open
    sidebar_open: rx.Field[bool] = rx.field(True)

    user_id: rx.Field[str] = rx.field(str(uuid.uuid4()))

    @rx.var
    def current_session(self) -> ChatSession | None:
        """Get the currently active session.

        Returns:
            ChatSession | None: The session matching `current_session_id`, if any.
        """
        for session in self.chat_sessions:
            if session.id == self.current_session_id:
                return session
        return None

    @rx.var
    def chat_history(self) -> list[tuple[str, str]]:
        """Get messages from the current session.

        Returns:
            list[tuple[str, str]]: Ordered list of (question, answer) message pairs.
        """
        if self.current_session:
            return self.current_session.messages
        return []

    @rx.var
    def sidebar_classes(self) -> str:
        """Generate sidebar CSS classes for the current visibility state.

        Returns:
            str: Tailwind-style class list controlling the sidebar presentation.
        """
        base = "bg-slate-2 border-r border-slate-5 transition-transform duration-300 ease-in-out z-40 overflow-hidden"
        transform = "translate-x-0" if self.sidebar_open else "-translate-x-full"
        return f"{transform} {base}"

    @rx.event
    def set_question(self, question: str):
        """Store the latest user prompt edit.

        Args:
            question: Text entered into the message input field.
        """
        self.question = question

    def create_new_session(self):
        """Create a new chat session and mark it active."""
        new_session = ChatSession(
            id=str(uuid.uuid4()),
            title="New Chat",
            created_at=datetime.now().strftime("%b %d, %Y %I:%M %p"),
            messages=[],
        )
        # Always assign a new list so Reflex registers the state change
        self.chat_sessions = [*self.chat_sessions, new_session]
        self.current_session_id = new_session.id

    @rx.event
    def switch_session(self, session_id: str):
        """Switch to a different session.

        Args:
            session_id: Identifier of the session that should become active.
        """
        # Ignore stale IDs that might arrive after a deletion event
        if not any(session.id == session_id for session in self.chat_sessions):
            if self.chat_sessions:
                self.current_session_id = self.chat_sessions[0].id
            else:
                self.current_session_id = ""
            return
        self.current_session_id = session_id

    @rx.event
    def delete_session(self, session_id: str):
        """Delete a specific session and recover the UI if needed.

        Args:
            session_id: Identifier of the session that should be removed.
        """
        remaining_sessions = [s for s in self.chat_sessions if s.id != session_id]
        self.chat_sessions = remaining_sessions

        # If we deleted the current session, switch to another or create new
        if self.current_session_id == session_id:
            if remaining_sessions:
                self.current_session_id = remaining_sessions[0].id
            else:
                self.create_new_session()

    @rx.event
    def toggle_sidebar(self):
        """Toggle sidebar visibility."""
        self.sidebar_open = not self.sidebar_open

    def update_session_title(self, first_question: str):
        """Update the title of the current session from the first prompt.

        Args:
            first_question: Text of the initial user message in the session.
        """
        if self.current_session:
            for session in self.chat_sessions:
                if session.id == self.current_session_id:
                    # Use first 50 chars of question as title
                    session.title = (
                        first_question[:50] + "..."
                        if len(first_question) > 50
                        else first_question
                    )
                    break

    @rx.event(background=True)
    async def answer(self):
        """Stream an LLM answer for the active question in the background.

        This method optimistically appends the user message, streams assistant text
        from the configured agent, and keeps the UI responsive while processing.
        Errors encountered during streaming are surfaced in the chat history.
        """
        # Validation and initial setup (must be done within async with self)
        async with self:
            if not self.question.strip():
                return

            # Create a new session if none exists
            if not self.current_session_id:
                self.create_new_session()

            # Store question and clear input
            question = self.question
            self.question = ""

            # Set processing state
            self.processing = True

            # Get current session ID before starting background work
            session_id = self.current_session_id

            # Add user message to current session
            for session in self.chat_sessions:
                if session.id == session_id:
                    session.messages.append((question, ""))
                    # Update title if this is the first message
                    if len(session.messages) == 1:
                        self.update_session_title(question)
                    break

        try:
            # Get current session messages for context
            async with self:
                current_messages = []
                if self.current_session:
                    current_messages = self.current_session.messages[
                        :-1
                    ]  # Exclude the current empty assistant message

            # Build conversation context (outside lock - no state access)
            conversation_context = ""
            for user_msg, assistant_msg in current_messages:
                conversation_context += f"User: {user_msg}\n"
                if assistant_msg:  # Skip empty assistant responses
                    conversation_context += f"Assistant: {assistant_msg}\n"

            # Add the current question
            full_prompt = f"{conversation_context}User: {question}\nAssistant:"

            # Initialize pydantic_ai agent
            agent = Agent("ollama:llama3.2")

            # Stream the response using run_stream
            accumulated_response = ""

            async with agent.run_stream(full_prompt) as result:
                # Stream text as it comes in from Ollama
                async for text_chunk in result.stream_text(
                    delta=False, debounce_by=0.05
                ):
                    # text_chunk contains the full text so far (delta=False)
                    accumulated_response = text_chunk

                    # Update the last message with streaming response (brief lock)
                    async with self:
                        for session in self.chat_sessions:
                            if session.id == session_id:
                                if session.messages:
                                    # Update the AI response in place
                                    session.messages[-1] = (
                                        question,
                                        accumulated_response,
                                    )
                                break

        except Exception as e:
            # Handle errors gracefully
            error_msg = f"Error: {str(e)}"
            async with self:
                for session in self.chat_sessions:
                    if session.id == session_id:
                        if session.messages:
                            session.messages[-1] = (
                                session.messages[-1][0],
                                error_msg,
                            )
                        break

        finally:
            # Always reset processing state
            async with self:
                self.processing = False

    @rx.event
    async def handle_key_down(self, key: str):
        """Dispatch background answering when Enter is pressed.

        Args:
            key: Keyboard key identifier from the input event.

        Returns:
            Callable | None: The `State.answer` event when submission should fire.
        """
        if key == "Enter" and not self.processing:
            # Trigger the background answer event
            return State.answer

    @rx.event
    def clear_chat(self):
        """Reset chat state by opening a fresh session."""
        # Create a new session (effectively clearing the current chat)
        self.create_new_session()
        self.processing = False
