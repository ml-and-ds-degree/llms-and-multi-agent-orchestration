import reflex as rx

from app.state import ChatSession, State


def chat_session_item(session: ChatSession) -> rx.Component:
    """Individual chat session item in the sidebar"""
    return rx.box(
        rx.hstack(
            # Session info
            rx.vstack(
                rx.text(
                    session.title,
                    font_weight="500",
                    class_name="text-sm text-slate-12",
                    white_space="nowrap",
                    overflow="hidden",
                    text_overflow="ellipsis",
                    width="100%",
                    # Critical: Allow text to shrink below content size in flex layout
                    min_width="0",
                ),
                rx.text(
                    session.created_at,
                    class_name="text-xs text-slate-9",
                ),
                align_items="start",
                spacing="1",
                width="100%",
                overflow="hidden",
                # Critical: Allow vstack to shrink below content size
                min_width="0",
            ),
            # Delete button
            rx.button(
                rx.icon("trash-2", size=16),
                on_click=lambda: State.delete_session(session.id),
                variant="ghost",
                size="1",
                color_scheme="red",
                class_name="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0",
            ),
            width="100%",
            align_items="center",
            justify_content="space-between",
            # Critical: Allow hstack to properly constrain children
            min_width="0",
            gap="2",
        ),
        # Make entire box clickable
        on_click=lambda: State.switch_session(session.id),
        # Styling
        class_name=rx.cond(
            State.current_session_id == session.id,
            "bg-accent-3 border-accent-6",
            "bg-transparent border-transparent hover:bg-slate-3",
        )
        + " px-3 py-2 rounded-lg cursor-pointer border transition-colors group",
        width="100%",
        overflow="hidden",
        # Fix border overflow by ensuring proper box sizing
        box_sizing="border-box",
    )


def sidebar() -> rx.Component:
    """The sidebar with chat history"""
    return rx.box(
        # Main sidebar content
        rx.vstack(
            # Header with close button
            rx.hstack(
                rx.heading("Chat History", size="5", class_name="text-slate-12"),
                # Close button (burger/menu icon)
                rx.button(
                    rx.icon("menu", size=18),
                    on_click=State.toggle_sidebar,
                    variant="ghost",
                    size="2",
                    class_name="cursor-pointer hover:bg-slate-4 active:bg-slate-5 transition-all rounded-md",
                    title="Close sidebar",
                    aria_label="Close sidebar",
                ),
                width="100%",
                align_items="center",
                justify_content="space-between",
            ),
            # List of chat sessions
            rx.scroll_area(
                rx.cond(
                    State.chat_sessions,
                    rx.vstack(
                        rx.foreach(
                            State.chat_sessions,
                            chat_session_item,
                        ),
                        width="100%",
                        spacing="2",
                    ),
                    rx.text(
                        "No chats yet. Start a new conversation!",
                        class_name="text-slate-10 text-sm text-center py-8",
                    ),
                ),
                type="auto",
                scrollbars="vertical",
                class_name="flex-1",
            ),
            width="100%",
            height="100%",
            spacing="4",
            class_name="p-4",
        ),
        # Sidebar container styling
        position="fixed",
        left="0",
        top="0",
        height="100vh",
        width="280px",
        class_name=rx.cond(
            State.sidebar_open,
            "translate-x-0",
            "-translate-x-full",
        )
        + " bg-slate-2 border-r border-slate-5 transition-transform duration-300 ease-in-out z-40 overflow-hidden",
    )


def sidebar_toggle_button() -> rx.Component:
    """Toggle button to show/hide sidebar when closed"""
    return rx.cond(
        ~State.sidebar_open,
        rx.button(
            rx.icon("menu", size=20),
            on_click=State.toggle_sidebar,
            variant="soft",
            size="2",
            position="fixed",
            left="1rem",
            top="1rem",
            z_index="50",
            class_name="cursor-pointer",
            title="Open sidebar",
        ),
    )
