import reflex as rx
from app.state import State
from app.components.hint import hint


def reset() -> rx.Component:
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
