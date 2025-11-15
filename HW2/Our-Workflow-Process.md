# Our Workflow Process

## Setup

We use [opencode](https://opencode.ai/) to which allows us to work with any provider. We chose GitHub Copilot because it supports non-token-consuming models, such as in our work gpt-5-mini.
![alt text](image.png)

## Agents

We created three agents for translation between English, French, and German using `opencode agent create`.

## Commands

We created a command to perform round-trip translation using cloude-sonnet 4.5 thought Github Copilot.

## Run

The orchestrated run can be seen [here](https://opencode.ai/s/UkKxcNUc)
