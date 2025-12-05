# Option 3 for Homework Assignment 5

## Lab Experiment: Installing Ollama RAG

### General Structure -- 4 Sub-experiments

### 1. Introduction

* This assignment deals with the analysis and study of installations based on RAG and OLLAMA .
* The series of experiments proposed below constitutes a general conceptual framework, and you are invited to interpret, develop, and explore the topics in any way you see fit .
* For each experiment, you must define research questions, perform the experiments, and analyze the findings, preferably by presenting a visual statistical analysis (using graphs or tables) .
* It is recommended to repeat each experiment several times to ensure statistical validity of the result .
* **Note:** Your conclusions do not have to overlap with the material presented in class. You are allowed to reach independent insights, provided you justify them well .
* In these cases, it is recommended to use external references and offer an explanation for the gaps you discovered .
* Take the experiments to a place that interests you and towards your personal direction of inquiry. These guidelines are intended to serve as 'brainstorming' and are not closed definitions .

### 2. Tools for Use in the Experiment

* This experiment is intended for technical installation enthusiasts .
* This experiment is based on the YouTube video: **Ollama Course -- Build AI Apps Locally**
  * Link: [https://www.youtube.com/watch?v=GWB9ApTPTv4&t=123s](https://www.youtube.com/watch?v=GWB9ApTPTv4&t=123s)
* In addition, an explanatory booklet written based on the video is attached .

### 3. Sub-experiment 1: "Baseline" (Correct according to the video)

**Goal:** To replicate 1:1 what the instructor does in the video to establish a "reference line" .

#### 1.3 The Student is Requested

**1.1.3 To Install and Run**

* A local server as default on Ollama (`localhost:11434`) .
* Basic language model: for example `llama3.2` or `llama 3.2 3B` (as recommended) .
* Embedding model: perform a pull for `nomic-embed-text` .

**2.1.3 To Build a "Classic" RAG as Described in the Book/Video**

* **Chunking:** Use `RecursiveCharacterTextSplitter` with `chunk_size=1200` and an overlap of approximately 300 characters ($\approx300$) .
* **Embeddings:** Use `OllamaEmbeddings` with `model='nomic-embed-text'` .
* **Vector Store:** Use Chroma .
  * Command: `Chroma.from_documents(..., embedding_function=OllamaEmbeddings, persist_directory=...)` .
* **Retrieval + Generation:** Perform `similarity_search(query, k=3)` .
* Prompt style: "Answer the question based ONLY on the following context: ..." .

**3.1.3 To Run a Short Python Script that Performs:**

* Indexing (one-time) of a small document (e.g., a short internal policy page provided to the student) .
* Asking 2-3 simple questions .
* Measuring :
  * Indexing time (seconds) .
  * Average answer time .
  * Is the answer accurate (Yes/No, according to a known solution) .

**4.1.3 The Student Must Verify Everything Follows the Video Instructions**

* The same `ollama pull / create / run` commands .
* The same embedding model .
* The same Vector Store library (Chroma) and the same integration (LangChain) .

#### 2.3 Output: Short "Baseline Report"

* System configuration (models, libraries, port, persist directory) .
* Run times .
* Answer quality .

### 4. Sub-experiment 2: Changing One Decision -- And What Breaks?

**Goal:** To understand installation and stack sensitivity: small change $\rightarrow$ technical/behavioral impact .

The student chooses **at least 2** of the following changes (based on what the video recommended *not* to do / did not do in practice) :

**1.4 Changing the LLM Model**

* Instead of `llama3.2`, a smaller model (3B) or larger (8B) .
* **Check:**
  * Response time (Latency) .
  * Memory usage (if measurable) .
  * Answer quality (same questions as Experiment 1) .

**2.4 Changing the Embedding Model**

* Instead of `nomic-embed-text`, another Embedding model (if the video/book mentions an alternative; or even a different Embedding) .
* **Check:**
  * Retrieval quality (do the same Chunks return?) .
  * Possible connection errors (server address, API key, network delay) .

**3.4 Changing Chunking Parameters**

* Very small `chunk_size` (e.g., 300) or very large (3000), different `overlap` .
* **Check:**
  * The Chunks created.
  * Answer quality: Was a "not found in context" situation created because information was cut/flooded? .

**4.4 "Forgetting" to Persist the Vector Store**

* Do not use `persist_directory`; every run builds the index anew .
* **Check:**
  * "Indexing" time in each run .
  * Sense of usability/Scalability (intuition regarding production) .

**5.4 Comparison Table**
The student re-runs the same queries from Experiment 1 and is requested to fill a table :

| Metric | Baseline | Change A | Change B |
| :--- | :--- | :--- | :--- |
| Average response time | | | |
| Number of Chunks | | | |
| Is the answer accurate | | | |
| Technical issues | | | |

**6.4 Pedagogical Goal**
To cause them to understand the technical implications of the recommendations in the video: why specifically this chunk size, why `nomic-embed-text`, why Chroma, and why separate Indexing from Retrieval .

### 5. Sub-experiment 3: Local Ollama vs. Free Cloud Alternative

**Goal:** To illustrate the differences between :

* "Local computational freedom" (Ollama on your laptop) .
* Free/cheap cloud service (e.g., public API with limits) .

The experiment can also be theoretical/simulative if you don't really want to connect an API, but a concrete code example is preferred .

#### 1.5 The Student is Requested

**1.1.5 To Replicate the RAG Pipeline**

* Replace the local LLM (`llama3.2` via Ollama) with a Cloud LLM (e.g., a free model on HuggingFace Inference, Groq, Together, etc.) .
* Keep all other stages local: Chunking, Embeddings, Vector Store .

**2.1.5 To Run the Exact Same Query in Three "Modes"**

1. **All Local Mode:** Ollama LLama + nomic-embed-text + Chroma .
2. **Hybrid Mode:** Local Embeddings, Cloud LLM .
3. **Cloud Mode (Optional):** Cloud Embeddings + Cloud LLM (if possible/desired) .

**3.1.5 To Measure**

* Latency (including network) .
* Network dependency (can it run offline?) .
* Authentication/Rate Limit faults .
* Answer quality (should be similar) .

#### 2.5 Desired Conclusions

* Understand the tradeoff: cost/time/privacy/installation simplicity .
* See that the architecture itself (RAG) is identical, only the "engine" changes .

### 6. Sub-experiment 4: "RAG Types" and Advanced Approaches

**Goal:** To link the theory from the chapters on RAG failure types and Context strategies -- to technical implementation .
The student will experiment with **at least two** variations:

**1.6 Basic RAG vs Contextual Retrieval**

* **1.1.6 Basic:** Perform Embedding for each Chunk as is .
* **2.1.6 Contextual Retrieval (inspired by Anthropic):** Before embedding, create a "title/context description" for each document (by LLM), attach the description to the Chunk, and then Embedding .
* **3.1.6 Comparison:**
  * Do "borderline" queries (e.g., implied phrasing) succeed more in the second method? .
  * Measurement: How many out of N queries return with the "correct" Chunk .

**2.6 Basic Retrieval vs Reranking**

* **1.2.6 Regular Retrieval:** Perform `similarity_search(k=5)` and throw directly to LLM .
* **2.2.6 Reranking:** Perform `similarity_search(k=10)` $\rightarrow$ use LLM (local or cloud) for re-ranking according to match with the question $\rightarrow$ insert only top 3 to LLM for the final question .
* **3.2.6 Check:** Are there fewer "Missed Top Rank / Not in Context" failures as described in the anti-patterns chapter .

**3.6 Goal of the Experiment**
To turn the theoretical text description (correct chunking, rerank, contextual) and the anti-patterns strategies into measurable code .

### 7. Meta-Pedagogical Requirements

To ensure the student actually watches the video:

**1.7 Video-Specific Questions**
In each sub-experiment, there is a "video-specific question", for example :

* "Which Embedding model is recommended in the video and why?" - They must identify this from the video/transcript .
* "What port is the Ollama server on and when should it be changed?" .
* "Which flags appear in the `ollama create` command in the demo?" .

**2.7 Final Checklist**
The student needs to attach:

* Screenshot/terminal output of ollama commands taken from the video .
* Short verbal explanation: "What was the technical decision the video proposed, and what did I do differently in sub-experiment 2/3/4" .

### 8. Experiment Summary

The experiment is short (small document, few queries) but rich in implementation variations .

**1.8 Each Sub-experiment**

* Uses the same basic pipeline (Ollama+LangChain+Chroma+Embeddings) .
* Changes one component/parameter and measures technical impact (Latency, errors, RAG failures) .

**2.8 The Result**
The student not only "knows what Ollama-RAG is", but understands :

* How installation/configuration influences things .
* What the local/cloud alternatives are .
* Why the recommendations in the video/book are what they are, and what is the cost of deviating from them .

**3.8 Presentation of Results**

* The student must plan and think about how to convincingly present the experiment results, the experiment inquiry, and the conclusions from the experiment .
* It is recommended to validate the results with graphs at the student's discretion .

**Note:** This document is written in the masculine form for convenience only, but is intended for both women and men .
