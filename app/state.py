import asyncio
import os
import uuid

import reflex as rx
from pydantic_ai import Agent

os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"


class SettingsState(rx.State):
    # The accent color for the app
    color: str = "violet"

    # The font family for the app
    font_family: str = "Poppins"

    @rx.event
    def set_color(self, color: str):
        self.color = color

    @rx.event
    def set_font_family(self, font_family: str):
        self.font_family = font_family


class State(rx.State):
    # The current question being asked.
    question: str = ""

    # Whether the app is processing a question.
    processing: bool = False

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]] = []

    user_id: str = str(uuid.uuid4())

    @rx.event
    def set_question(self, question: str):
        self.question = question

    async def answer(self):
        # Set the processing state to True.
        self.processing = True
        yield

        # convert chat history to a list of dictionaries
        chat_history_dicts = []
        for chat_history_tuple in self.chat_history:
            chat_history_dicts.append(
                {"role": "user", "content": chat_history_tuple[0]}
            )
            chat_history_dicts.append(
                {"role": "assistant", "content": chat_history_tuple[1]}
            )

        self.chat_history.append((self.question, ""))

        # Clear the question input.
        question = self.question
        self.question = ""

        # Yield here to clear the frontend input before continuing.
        yield

        # Initialize pydantic_ai agent
        agent = Agent("ollama:llama3.2")

        # Build the full conversation context
        conversation_context = ""
        for msg in chat_history_dicts:
            if msg["role"] == "user":
                conversation_context += f"User: {msg['content']}\n"
            else:
                conversation_context += f"Assistant: {msg['content']}\n"

        # Add the current question
        full_prompt = f"{conversation_context}User: {question}\nAssistant:"

        # Run the agent
        result = await agent.run(full_prompt)
        answer = result.output

        for i in range(len(answer)):
            # Pause to show the streaming effect.
            await asyncio.sleep(0.01)
            # Add one letter at a time to the output.
            self.chat_history[-1] = (
                self.chat_history[-1][0],
                answer[: i + 1],
            )
            yield

        # Set the processing state to False.
        self.processing = False

    async def handle_key_down(self, key: str):
        if key == "Enter":
            async for t in self.answer():
                yield t

    def clear_chat(self):
        # Reset the chat history and processing state
        self.chat_history = []
        self.processing = False
