# llms-and-multi-agent-orchestration

An interactive chat experience built with [Reflex](https://reflex.dev/) on the frontend and [pydantic-ai](https://ai.pydantic.dev/latest/) orchestrating an [Ollama](https://ollama.com/) language model on the backend. The UI lets end users converse with a local LLaMA 3.2 model, stream responses in real time, and fine-tune the look and feel directly from the browser.

---

## ✨ Highlights

- **Responsive Reflex UI** with keyboard support, streaming answer animation, and a polished send button.
- **Live customization** of theme (system / light / dark) and font family via settings popover.
- **Memoryful conversations** – chat history is preserved in state and reused in prompts for richer answers.
- **Local-first LLM orchestration** using `pydantic_ai` and Ollama, keeping data on your machine.
- **Both UI & CLI entry points** for fast testing (`main.py`) or full experience (`reflex run`).

---

## 🗂 Project Structure

```bash
.
├── app/
│   ├── app.py              # Reflex app definition (root component, routing, themes)
│   ├── state.py            # Chat + settings state machines powering the UI
│   ├── style.py            # Shared design tokens used across the app
│   ├── components/         # Sidebar, settings popover, buttons, templates, etc.
│   └── views/              # Page-level layout primitives (chat area, action bar)
├── assets/                 # Static assets served by Reflex (icons, SVG logos)
├── prompts.md              # Prompt engineering notes and reusable templates
├── tests/
│   ├── test_state.py       # Unit tests for the Reflex state logic
│   ├── e2e/                # Scenario placeholders + helper scripts
│   └── setup_tests.sh      # Bootstraps test environment locally/CI
├── pyproject.toml          # Project metadata and Python dependencies (via uv)
├── rxconfig.py             # Reflex configuration (app name, API URL, etc.)
├── pytest.ini              # Pytest defaults for the test suite
├── uv.lock                 # Resolved dependency lockfile for reproducible installs
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

The UI will be available at <http://localhost:3000> (default Reflex port).

Key interactions:

- Type in the prompt bar or hit **Enter** to submit.
- Click the **Send** button to dispatch your prompt (spinner shows while the agent responds).
- Use the **Settings** icon (top right) to tweak fonts and light/dark/system theme.
- Select a template card to quickly populate and ask a sample question.
- Hit **New Chat** to clear your conversation history.

---

## ⚙️ Configuration

| Setting                    | Description                                    | Default                     |
| -------------------------- | ---------------------------------------------- | --------------------------- |
| `OLLAMA_BASE_URL`          | Endpoint for the Ollama REST API               | `http://localhost:11434/v1` |
| `Agent("ollama:llama3.2")` | Model identifier passed to `pydantic_ai.Agent` | `ollama:llama3.2`           |
| `SettingsState` fields     | Font family (`Poppins`)                        | Change at runtime via UI    |

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

## 🤖 OpenCode Setup

We've chosen to focus on [OpenCode](https://opencode.ai/) as our primary development tool for this project. OpenCode is an AI coding agent for the terminal that supports **multiple LLM providers** (OpenAI, Anthropic, GitHub Copilot, Ollama, and more), making it flexible and provider-agnostic. This allows the team to leverage AI assistance for building features, debugging issues, and understanding the codebase without being locked into a single provider.

### Specialized Subagents

Two framework-expert subagents are configured in `.opencode/agent/`, both using **GitHub Copilot's non-token-consuming model** (GPT-5 mini) for extremely cost-effective assistance:

**`@reflex-docs-expert`** – Retrieves Reflex framework documentation and best practices

**`@pydantic-ai-expert`** – Retrieves Pydantic AI documentation and implementation guidance
