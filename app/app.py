import reflex as rx

from app import style
from app.components.reset import reset
from app.components.settings import settings_icon
from app.components.sidebar import sidebar, sidebar_toggle_button
from app.state import SettingsState
from app.views.chat import action_bar, chat
from app.views.templates import templates


def index() -> rx.Component:
    return rx.theme(
        rx.el.style(
            f"""
            :root {{
                --font-family: "{SettingsState.font_family}", sans-serif;
            }}
        """
        ),
        # Sidebar
        sidebar(),
        # Toggle button when sidebar is closed
        sidebar_toggle_button(),
        # Top bar with the reset and settings buttons
        rx.box(
            reset(),
            settings_icon(),
            class_name="top-4 right-4 absolute flex flex-row items-center gap-3.5 z-30",
        ),
        # Main content - always centered regardless of sidebar
        rx.box(
            # Prompt examples
            templates(),
            # Chat history
            chat(),
            # Action bar
            action_bar(),
            class_name="relative flex flex-col justify-between gap-20 mx-auto px-6 pt-16 lg:pt-6 pb-6 max-w-4xl h-screen",
        ),
        accent_color="violet",
    )


app = rx.App(stylesheets=style.STYLESHEETS, style={"font_family": "var(--font-family)"})
app.add_page(
    index, title="Chatbot", description="A chatbot powered by Reflex and LlamaIndex!"
)
