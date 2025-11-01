import reflex as rx

from app.state import State


def qa(question: str, answer: str) -> rx.Component:
    return rx.box(
        # User question bubble
        rx.box(
            rx.markdown(
                question,
                class_name="[&>p]:!my-2.5",
            ),
            padding="0.65rem 1rem",
            border_radius="1.25rem 1.25rem 0.25rem 1.25rem",
            max_width="70%",
            background=rx.color("blue", 9),
            color=rx.color("gray", 1),
            box_shadow="0 2px 8px rgba(0, 0, 0, 0.08)",
            align_self="flex-end",
            transition="all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
            _hover={
                "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.12)",
                "transform": "translateY(-1px)",
            },
            word_break="break-word",
        ),
        # Assistant answer with avatar
        rx.box(
            # Avatar
            rx.box(
                rx.image(
                    src="llama.svg",
                    height="1.5rem",
                    class_name=rx.cond(State.processing, " animate-pulse", ""),
                ),
                flex_shrink="0",
                margin_top="0.25rem",
            ),
            # Answer bubble
            rx.box(
                # Show animated dots when answer is empty (thinking), otherwise show the answer
                rx.cond(
                    answer == "",
                    # Animated typing dots
                    rx.html(
                        """
                        <style>
                            @keyframes dot-bounce {
                                0%, 80%, 100% { opacity: 0.3; transform: translateY(0); }
                                40% { opacity: 1; transform: translateY(-4px); }
                            }
                            .typing-dots {
                                display: flex;
                                align-items: center;
                                gap: 3px;
                                height: 20px;
                            }
                            .typing-dot {
                                width: 6px;
                                height: 6px;
                                border-radius: 50%;
                                background-color: currentColor;
                                animation: dot-bounce 1.4s infinite ease-in-out;
                            }
                            .typing-dot:nth-child(1) { animation-delay: 0s; }
                            .typing-dot:nth-child(2) { animation-delay: 0.2s; }
                            .typing-dot:nth-child(3) { animation-delay: 0.4s; }
                        </style>
                        <div class="typing-dots">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                        """
                    ),
                    # Actual answer content
                    rx.markdown(
                        answer,
                        class_name="[&>p]:!my-2.5",
                    ),
                ),
                # Copy button (only show when answer is not empty)
                rx.cond(
                    answer != "",
                    rx.box(
                        rx.el.button(
                            rx.icon(tag="copy", size=18),
                            on_click=[rx.set_clipboard(answer), rx.toast("Copied!")],
                            title="Copy message",
                            style={
                                "padding": "0.25rem",
                                "color": rx.color("gray", 10),
                                "background": "transparent",
                                "border": "none",
                                "cursor": "pointer",
                                "border_radius": "0.25rem",
                                "transition": "all 0.15s ease",
                            },
                            _hover={
                                "color": rx.color("gray", 12),
                                "background": rx.color("gray", 4),
                            },
                        ),
                        position="absolute",
                        bottom="-2.25rem",
                        left="1.25rem",
                        opacity="0",
                        _group_hover={"opacity": "1"},
                        transition="opacity 0.2s ease",
                    ),
                ),
                position="relative",
                padding="0.65rem 1rem",
                border_radius="1.25rem 1.25rem 1.25rem 0.25rem",
                max_width="70%",
                background=rx.color("accent", 4),
                color=rx.color("gray", 12),
                box_shadow="0 2px 8px rgba(0, 0, 0, 0.06)",
                border=f"1px solid {rx.color('accent', 6)}",
                align_self="flex-start",
                transition="all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                _hover={
                    "box_shadow": "0 4px 12px rgba(0, 0, 0, 0.1)",
                    "border_color": rx.color("accent", 7),
                },
                word_break="break-word",
            ),
            display="flex",
            flex_direction="row",
            gap="1.5rem",
            align_items="flex-start",
        ),
        display="flex",
        flex_direction="column",
        gap="2rem",
        padding_bottom="2.5rem",
        class_name="group",
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
                class_name=(
                    "box-border bg-slate-2 px-4 py-2 rounded-full w-full outline-none focus:outline-[#6E56CF] "
                    "h-[48px] text-slate-12 placeholder:text-slate-9 border border-slate-5 focus:border-[#6E56CF] "
                    "shadow-sm focus:shadow-md transition-colors transition-shadow"
                ),
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
    )
