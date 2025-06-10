# Progress Bar Messages
MSG_SPLITTING = "📄 Splitting document..."
MSG_CREATING_VECTORSTORE = "📌 Creating vectorstore..."
MSG_RUNNING_GRAPH = "🤖 Running graph inference..."
MSG_SAVING_RESULT = "💾 Saving result..."
MSG_DONE = "✅ Done"

# Constants
OPENAI_MODEL_NAME = None  # gpt-3.5-turbo by default
HUGGINGFACE_MODEL_NAME = "intfloat/multilingual-e5-large"  #

# UI Labels & Messages
PAGE_TITLE = "AI Document Analyzer"
APP_TITLE = "📄 AI-Powered Document Analyzer (LangChain + LangGraph + RAG)"
BUTTON_REGENERATE = "♻️ Regenerate Analysis"
TOGGLE_RAW = "🛠 Show raw output"
OPEN_AI_TITLE = (
    f"💡 Model used: **{OPENAI_MODEL_NAME or 'OpenAI GPT-3.5 Turbo'}** (via API)"
)
OLLAMA_TITLE = "💡 Model used: **Ollama Mistral** (local build via Ollama)"

# Allowed ile Types
FILE_TYPES = ["pdf"]

# Section Headers
SECTION_SUMMARY = "🔍 Summary"
SECTION_QA = "🧠 Answers to Predefined Questions"

# Progress Messages
MSG_DONE = "✅ Loaded from cache"
MSG_PROCESSING = "Processing... This might take a minute ⏳"
