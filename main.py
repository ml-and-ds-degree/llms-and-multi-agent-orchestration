import os

from pydantic_ai import Agent

os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"

agent = Agent("ollama:llama3.2")

print(agent.run_sync("Hello, who are you?").output)
