---
description: Translate English through French and German and back
---

Please perform a round-trip translation test on the following English sentence: "$ARGUMENTS"

Follow these steps:

1. First, translate the English sentence to French using the @english-french-translator agent
2. Then, take the French translation and translate it to German using the @french-german-translator agent  
3. Finally, take the German translation and translate it back to English using the @german-english-translator agent

Show me:
- Original English: $ARGUMENTS
- English → French translation
- French → German translation  
- German → English translation
- Analysis of how the meaning changed (if at all) through the round-trip translation
