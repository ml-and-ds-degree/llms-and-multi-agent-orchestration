import reflex as rx

from app.components.badge import made_with_reflex
from app.state import State


def qa(question: str, answer: str) -> rx.Component:
    return rx.box(
        # Question
        rx.box(
            rx.markdown(
                question,
                class_name="[&>p]:!my-2.5",
            ),
            class_name="relative bg-slate-3 px-5 rounded-3xl max-w-[70%] text-slate-12 self-end",
        ),
        # Answer
        rx.box(
            rx.box(
                rx.image(
                    src="llama.svg",
                    class_name="h-6" + rx.cond(State.processing, " animate-pulse", ""),
                ),
            ),
            rx.box(
                rx.markdown(
                    answer,
                    class_name="[&>p]:!my-2.5",
                ),
                rx.box(
                    rx.el.button(
                        rx.icon(tag="copy", size=18),
                        class_name="p-1 text-slate-10 hover:text-slate-11 transform transition-colors cursor-pointer",
                        on_click=[rx.set_clipboard(answer), rx.toast("Copied!")],
                        title="Copy",
                    ),
                    class_name="-bottom-9 left-5 absolute opacity-0 group-hover:opacity-100 transition-opacity",
                ),
                class_name="relative bg-accent-4 px-5 rounded-3xl max-w-[70%] text-slate-12 self-start",
            ),
            class_name="flex flex-row gap-6",
        ),
        class_name="flex flex-col gap-8 pb-10 group",
    )


def chat() -> rx.Component:
    return rx.scroll_area(
        rx.foreach(
            State.chat_history,
            lambda messages: qa(messages[0], messages[1]),
        ),
        scrollbars="vertical",
        class_name="w-full",
    )


def action_bar() -> rx.Component:
    return rx.box(
        rx.box(
            rx.el.input(
                placeholder="Ask anything",
                value=State.question,
                on_change=State.set_question,
                on_key_down=State.handle_key_down,
                id="input1",
                class_name="box-border bg-slate-3 px-4 py-2 rounded-full w-full outline-none focus:outline-accent-10 h-[48px] text-slate-12 placeholder:text-slate-9",
            ),
            rx.button(
                rx.cond(
                    State.processing,
                    rx.icon(
                        tag="loader-circle",
                        size=18,
                        class_name="animate-spin",
                    ),
                    rx.hstack(
                        rx.icon(tag="send", size=18),
                        rx.text("Send", class_name="text-base font-semibold"),
                        spacing="2",
                        class_name="items-center",
                    ),
                ),
                on_click=[State.answer, rx.set_value("input1", "")],
                class_name=(
                    "bg-[#6E56CF] hover:bg-[#5A46B8] disabled:hover:bg-[#6E56CF] text-white "
                    "px-6 h-[48px] rounded-full transition-colors flex items-center justify-center "
                    "cursor-pointer disabled:cursor-default shadow-md shadow-[#6E56CF]/40 disabled:opacity-60"
                ),
                disabled=rx.cond(
                    State.processing | (State.question == ""), True, False
                ),
            ),
            class_name="flex flex-row gap-3 w-full",
        ),
        # Made with Reflex link
        made_with_reflex(),
        class_name="flex flex-col justify-center items-center gap-6 w-full",
    )
