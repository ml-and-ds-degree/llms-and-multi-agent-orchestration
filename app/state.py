import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import reflex as rx
from pydantic_ai import Agent

os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"


class SettingsState(rx.State):
    # The font family for the app
    font_family: rx.Field[str] = rx.field("Poppins")

    @rx.event
    def set_font_family(self, font_family: str):
        self.font_family = font_family


@dataclass
class ChatSession:
    """Represents a single chat session"""

    id: str
    title: str
    created_at: str
    messages: list[tuple[str, str]] = field(default_factory=list)


class State(rx.State):
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
        """Get the currently active session"""
        for session in self.chat_sessions:
            if session.id == self.current_session_id:
                return session
        return None

    @rx.var
    def chat_history(self) -> list[tuple[str, str]]:
        """Get messages from current session"""
        if self.current_session:
            return self.current_session.messages
        return []

    @rx.var
    def sidebar_classes(self) -> str:
        """Generate sidebar CSS classes based on state"""
        base = "bg-slate-2 border-r border-slate-5 transition-transform duration-300 ease-in-out z-40 overflow-hidden"
        transform = "translate-x-0" if self.sidebar_open else "-translate-x-full"
        return f"{transform} {base}"

    @rx.event
    def set_question(self, question: str):
        self.question = question

    def create_new_session(self):
        """Create a new chat session"""
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
        """Switch to a different session"""
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
        """Delete a specific session"""
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
        """Toggle sidebar visibility"""
        self.sidebar_open = not self.sidebar_open

    def update_session_title(self, first_question: str):
        """Update the title of current session based on first question"""
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

    @rx.event
    async def answer(self):
        """
        Process user question and get response from LLM.
        Uses yield to keep UI responsive during processing.
        """
        # Validation
        if not self.question.strip():
            return

        # Create a new session if none exists
        if not self.current_session_id:
            self.create_new_session()
            yield  # Update UI after creating session

        # Store question and clear input
        question = self.question
        self.question = ""

        # Set processing state
        self.processing = True

        # Add user message to current session
        for session in self.chat_sessions:
            if session.id == self.current_session_id:
                session.messages.append((question, ""))
                # Update title if this is the first message
                if len(session.messages) == 1:
                    self.update_session_title(question)
                break

        # Yield to show user message immediately
        yield

        try:
            # Get current session messages for context
            current_messages = []
            if self.current_session:
                current_messages = self.current_session.messages[
                    :-1
                ]  # Exclude the current empty assistant message

            # Build conversation context
            conversation_context = ""
            for user_msg, assistant_msg in current_messages:
                conversation_context += f"User: {user_msg}\n"
                if assistant_msg:  # Skip empty assistant responses
                    conversation_context += f"Assistant: {assistant_msg}\n"

            # Add the current question
            full_prompt = f"{conversation_context}User: {question}\nAssistant:"

            # Initialize pydantic_ai agent
            agent = Agent("ollama:llama3.2")

            # Run the agent
            result = await agent.run(full_prompt)
            answer = result.output

            # Stream the response character by character
            for session in self.chat_sessions:
                if session.id == self.current_session_id:
                    for i in range(len(answer)):
                        # Pause to show the streaming effect
                        await asyncio.sleep(0.01)
                        # Add one letter at a time to the output
                        session.messages[-1] = (
                            session.messages[-1][0],
                            answer[: i + 1],
                        )
                        yield
                    break

        except Exception as e:
            # Handle errors gracefully
            error_msg = f"Error: {str(e)}"
            for session in self.chat_sessions:
                if session.id == self.current_session_id:
                    session.messages[-1] = (
                        session.messages[-1][0],
                        error_msg,
                    )
                    break
            yield

        finally:
            # Always reset processing state
            self.processing = False
            yield

    @rx.event
    async def handle_key_down(self, key: str):
        if key == "Enter":
            async for t in self.answer():
                yield t

    @rx.event
    def clear_chat(self):
        # Create a new session (effectively clearing the current chat)
        self.create_new_session()
        self.processing = False
