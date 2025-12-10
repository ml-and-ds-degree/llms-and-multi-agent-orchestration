"""UI component for resetting the chat conversation."""

import reflex as rx

from app.components.hint import hint
from app.state import State


def reset() -> rx.Component:
    """Render the top-bar new chat action.

    Returns:
        rx.Component: Hoverable button that starts a fresh chat session.
    """
    return hint(
        text="New Chat",
        content=rx.box(
            rx.icon(
                tag="square-pen",
                size=22,
                stroke_width="1.5",
                class_name="!text-slate-10",
            ),
            class_name="p-2 rounded-xl cursor-pointer",
            on_click=State.clear_chat,
        ),
        side="bottom",
    )
