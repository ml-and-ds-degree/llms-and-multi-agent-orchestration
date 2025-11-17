# Our Workflow Process

## Setup

We use [opencode](https://opencode.ai/) to which allows us to work with any provider. We chose GitHub Copilot because it supports non-token-consuming models, such as in our work gpt-5-mini.
![Github Copilot models selection](./assets/image.png)

## Agents

We created three agents for translation between English, French, and German using `opencode agent create`.

## Commands

We created a command to perform round-trip translation using cloude-sonnet 4.5 thought Github Copilot.

## Run

The orchestrated run can be seen [here](https://opencode.ai/s/B26j4ji1)

## Sentence Metadata

We created three datasets with varying spelling error rates (0%, 25%, and 50%) containing the same 5 sentences. Below are the statistics for each sentence:

| Sentence | Word Count | Character Count | Topic |
|----------|-----------|------------------|-------|
| 1 | 19 | 161 | Artificial intelligence and financial data analysis |
| 2 | 22 | 174 | Mountain hiking and meteorological warnings |
| 3 | 21 | 176 | Quantum mechanics research |
| 4 | 23 | 190 | Sustainable economic growth and climate policy |
| 5 | 23 | 183 | Renewable energy technology development |

**Total Statistics:**

- Total sentences: 5
- Average words per sentence: 21.6
- Average characters per sentence: 176.8
- Total words across all sentences: 108
- Total characters across all sentences: 884
