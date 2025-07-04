# AI-Powered PDF Document Analyzer

This project allows you to upload and analyze long PDF documents using either OpenAI or local LLM Ollama. It performs summarization and question answering using LangChain, LangGraph, FAISS, and RAG architecture.

---

## Features

- Summarizes complex PDF documents
- Answers predefined questions based on document content
- Switches between OpenAI and local Ollama model
- Caches results for faster re-use

---

## Installation

### 1. Clone the Repository

```bash
git https://github.com/Oskaras-coder/homework-ai-document-analyzer.git
cd ai-document-analyzer
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## LLM Configuration

You can choose between using OpenAI or Ollama (local model).

### Option 1: Using OpenAI _(recommended for speed and accuracy)_

#### Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:

1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.

#### Enter the OpenAI API key

1. Create a `.env` file in the root directory and add your OpenAI API key:

```ini
OPENAI_API_KEY='xxxxxxxxxx'
```

2. (Optional) Set your desired OpenAI model in utils/constants.py:

```ini
OPENAI_MODEL_NAME = "gpt-3.5-turbo"
```

### Option 2: Using Ollama + Hugging Face Embeddings _(Free but slower)_

> Ollama model ~2 GB download  
> Hugging Face model ~4 GB download

1. Download and install Ollama for your OS. Go to:
   https://ollama.com/download
2. Run the Ollama server:

```bash
ollama serve
```

3. (Optional) Pull mistral

```bash
ollama pull mistral
```

4. (Optional) Customize model name in utils/constants.py:

```ini
HUGGINGFACE_MODEL_NAME = "intfloat/multilingual-e5-large"
```

---

## Running the App

Launch the Streamlit app:

```bash
streamlit run app.py
```

---

## Running Tests

To validate application integrity:

```bash
pytest test_app.py
```

---

## Notes

- JSON results are cached in `/results/` folder. They can also be read in the UI as raw material.
- Supports summaries of large (100+ pages) PDFs
- If no OpenAI key is provided, it automatically falls back to Ollama + Hugging Face
- Summary and QA generation speed Based on provided dataset:
  - Tallink Grupp Sustainability Reports
    - OPENAI GPT-3.5 Turbo - ~4 - 5 mins
    - Ollama - ~10 - 15 mins or longer
  - Eesti Energia Annual Report 2023 ENG
    - OPENAI GPT-3.5 Turbo - ~6 - 7 mins
    - Ollama - ~18 - 25 mins or longer

## Why these LLMs were used?

### OpenAI API

#### Pros

- Performance speed: Fastiest and highest-quality AI for the fast project implementation.

- Quality: Access to the very latest model improvements (GPT-4, GPT-3.5 Turbo, etc.) without any local maintenance.

- Scalability: Pay per token, your local machine’s specs don’t matter.

#### Cons

- Online-only: Requires a live internet connection and a valid API key.

- Cost: Billed by usage.

- Data Privacy: Your documents must be sent to OpenAI’s servers, if you have sensitive data, that may be undesirable. ✅ This project used open-source data.

### Ollama

#### Pros

- Offline Capability: Once a model (e.g. Mistral) is pulled locally, no internet is needed. Good choice for privacy-sensitive environments.

- Zero API Cost: You pay only in disk space (≈2 GB for Mistral plus ≈4 GB for the default Hugging Face embeddings), and CPU/GPU cycles on your own hardware.

- Control: You can swap in any compatible .gguf or LLaMA-format model, fine-tune it, or run proprietary in-house models.

#### Cons

- Speed: Local inference is generally 10× to 100× slower than a managed GPU API. Summarizing 100 pages can take several minutes.

- Setup Overhead: Users must install Ollama (or a similar runtime), pull down multi-GB model files, and have at least 8–16 GB of RAM.

- Model Updates: You’re responsible for updating and maintaining the model binaries yourself.

## Further Improvements & Limitations

This AI-powered document analyzer provides a solid foundation, but there are several areas that could be enhanced for greater functionality, performance, and reliability.

---

### Potential Improvements

#### Improved Testing and Exception Handling

- Extend the current test coverage using `pytest`, including edge cases and regression testing.
- Introduce robust exception handling across all modules (e.g., loaders, embeddings, summarizer) to make the app more resilient and easier to debug.

#### Integration and Benchmarking of More Models

- Incorporate additional models (e.g., Claude, Gemini, LLaMA2, Falcon).
- Evaluate their performance on real-world documents to determine which yields the most accurate and cost-efficient responses.

#### Multi-Document Support and Cross-Analysis

- Allow users to upload multiple PDFs for comparison.
- Generate summaries per file and then aggregate them to enable broader document analysis.

#### Sensitive Data Support

As the tool currently stores files locally, it is not suitable for handling confidential data. Improvements may include:

- In-memory-only processing
- Local LLMs with no internet access (for example further improve Ollama implementation)
- File encryption
- On-premise deployment via Docker

#### Enhanced User Interface

- Add features like:
  - Side-by-side view of PDF and extracted content
  - Highlighted text from source pages
- Reduce Chunk Size and Overlap
  - Reduce the number and size of chunks, speeding up both vectorization and generation but this comes with a risk of quality and should be tested.
