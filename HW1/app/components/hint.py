"""Reusable tooltip-style hover card helpers."""

import reflex as rx


def hint(
    text: str,
    content: rx.Component,
    side: str = "top",
    align: str = "center",
    active: bool = False,
    class_name: str = "",
    **props,
) -> rx.Component:
    """Wrap a component with a hover card showing helper text.

    Args:
        text: Tooltip content to display inside the card.
        content: Interactive element that triggers the hover card.
        side: Preferred side of the trigger to place the hover card.
        align: Alignment of the card relative to the trigger.
        active: Whether the card stays open by default.
        class_name: Optional CSS classes applied to the root element.
        **props: Additional keyword props forwarded to `hover_card.root`.

    Returns:
        rx.Component: Configured hover card component.
    """
    return rx.hover_card.root(
        rx.hover_card.trigger(content, height="fit-content"),
        rx.hover_card.content(
            rx.text(text),
            side=side,
            align=align,
            class_name="flex justify-center items-center bg-slate-11 px-1.5 py-0.5 rounded-lg text-[#000000] text-sm",
        ),
        class_name=class_name,
        default_open=active,
        open_delay=80,
        close_delay=80,
        **props,
    )
