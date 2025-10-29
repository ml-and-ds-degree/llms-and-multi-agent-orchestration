# llms-and-multi-agent-orchestration

An interactive chat experience built with [Reflex](https://reflex.dev/) on the frontend and [pydantic-ai](https://ai.pydantic.dev/latest/) orchestrating an [Ollama](https://ollama.com/) language model on the backend. The UI lets end users converse with a local LLaMA 3.2 model, stream responses in real time, and fine-tune the look and feel directly from the browser.

---

## ✨ Highlights
- **Responsive Reflex UI** with keyboard support, streaming answer animation, and a polished send button.
- **Live customization** of accent color, theme (system / light / dark), and font family via settings popover.
- **Memoryful conversations** – chat history is preserved in state and reused in prompts for richer answers.
- **Local-first LLM orchestration** using `pydantic_ai` and Ollama, keeping data on your machine.
- **Both UI & CLI entry points** for fast testing (`main.py`) or full experience (`reflex run`).

---

## 🗂 Project Structure

```
.
├── app/
│   ├── app.py              # Reflex app definition (theme, layout)
│   ├── state.py            # Core chat & settings state machines
│   ├── components/         # Reusable UI pieces (send button, reset, settings, badges)
│   └── views/              # Page-level layout (chat area, action bar, templates)
├── assets/                 # Static assets served by Reflex (icons, logos)
├── main.py                 # Minimal CLI utility to send prompts to the agent
├── pyproject.toml          # Project metadata and dependencies (Python ≥3.13)
└── README.md               # You're here
```

---

## 🚀 Prerequisites

- **Python 3.13** (Reflex currently targets 3.9–3.11, but the project uses 3.13 via [`uv`](https://github.com/astral-sh/uv); see troubleshooting notes below).
- **uv** package manager (recommended) – install from the official docs: `pip install uv`.
- **Ollama** running locally with the `llama3.2` model available:  
  ```bash
  brew install ollama  # or follow instructions at https://ollama.com/
  ollama pull llama3.2
  ollama serve  # ensures API at http://localhost:11434/v1
  ```
  > Want to point at another Ollama instance or model? Set `OLLAMA_BASE_URL` or adjust `Agent("ollama:...")` in `app/state.py`.

---

## 🧱 Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/llms-and-multi-agent-orchestration.git
cd llms-and-multi-agent-orchestration

# Create or refresh the virtual environment & install deps.
uv sync
```

`uv sync` creates `.venv/` (if missing) and installs the pinned `reflex` and `pydantic-ai[ollama]` versions from `pyproject.toml`.

---

## 💬 Running the Reflex App

Start Ollama (if not already running), then launch the Reflex development server through `uv` so it picks up the project environment:

```bash
uv run reflex run
```

The UI will be available at http://localhost:3000 (default Reflex port).  

Key interactions:
- Type in the prompt bar or hit **Enter** to submit.
- Click the **Send** button to dispatch your prompt (spinner shows while the agent responds).
- Use the **Settings** icon (top right) to tweak accent colors, fonts, and light/dark/system theme.
- Select a template card to quickly populate and ask a sample question.
- Hit **New Chat** to clear your conversation history.

---

## 🛠 CLI Prompting (Optional)

Prefer a fast terminal round-trip? Use the bundled helper script:

```bash
uv run python main.py "Explain multi-agent orchestration in simple terms."
```

If you omit the argument, the script will prompt you interactively.

---

## ⚙️ Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `OLLAMA_BASE_URL` | Endpoint for the Ollama REST API | `http://localhost:11434/v1` |
| `Agent("ollama:llama3.2")` | Model identifier passed to `pydantic_ai.Agent` | `ollama:llama3.2` |
| `SettingsState` fields | Accent color (`violet`), font family (`Poppins`) | Change at runtime via UI |

> To persist a different model or base URL, edit `app/state.py`. Reflex automatically reloads on save during development.

---

## 🔍 Troubleshooting

- **`ModuleNotFoundError: No module named 'pydantic_ai'`**  
  Make sure you run commands through `uv run …` or activate the project virtualenv before invoking `reflex run`.

- **Reflex expects Python ≤3.11**  
  This project pins Python ≥3.13. Using `uv` isolates interpreter management so the packages remain compatible. If you install dependencies manually, ensure the interpreter matches and reinstall Reflex with `pip install "reflex>=0.8.17"`.

- **Ollama not reachable**  
  Confirm the server is running (`ollama serve`) and that `OLLAMA_BASE_URL` matches. Try `curl http://localhost:11434/v1/models`.

- **Agent responses stall**  
  Check Ollama logs for request errors. The UI streams responses one character at a time; if the agent returns nothing, the stream will appear empty.

---

## 🧪 Development Tips

- Reflex hot reload keeps UI changes instant; state changes in Python files rebuild automatically.
- For faster debugging, instrument `State.answer` with logging (e.g., `print`) — Reflex pipes stdout to the console.
- Add integration tests around `State.answer` using `pytest` + `anyio` if you need coverage for async state transitions.

---

## 📄 License

Specify your licensing terms here (MIT, Apache 2.0, proprietary, etc.). Update this section to match your project’s requirements.

---

Happy building! Feel free to extend the components in `app/components/` or drop in additional agents via pydantic-ai’s multi-step tools. If you run into issues or have feature ideas, open an issue or submit a PR. 👋
