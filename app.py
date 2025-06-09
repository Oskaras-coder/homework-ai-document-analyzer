import os
import json
import streamlit as st  # Streamlit web app framework
from main import process_document
from utils.utils import get_file_hash
from utils.constants import (
    OLLAMA_TITLE,
    OPEN_AI_TITLE,
    PAGE_TITLE,
    APP_TITLE,
    FILE_TYPES,
    BUTTON_REGENERATE,
    SECTION_SUMMARY,
    SECTION_QA,
    TOGGLE_RAW,
    MSG_DONE,
    MSG_PROCESSING,
)
from core.llm import OPENAI_API_KEY


def _get_hash_and_path(file):  # Create unique hash & path
    file_hash = get_file_hash(file)  # Get unique hash of file contents
    return (
        file_hash,
        f"results/{file_hash}.json",
    )  # Return file hash and path to cache JSON


def _init_progress_bar():  # Set up progress UI
    progress_text = st.empty()
    progress_bar = st.progress(0)

    def on_progress(pct, message):
        progress_text.text(message)
        progress_bar.progress(pct)

    return progress_text, progress_bar, on_progress


def _clear_progress(text_widget, bar_widget):  # Clear progress UI
    text_widget.empty()
    bar_widget.empty()


def _load_cached_results(path):  # Load from cache
    with open(path, "r", encoding="utf-8") as f:
        cached = json.load(f)
    return cached["summary"], cached["qa"]


def _render_output(summary, answers):  # Display summary and Q&A
    show_raw = st.toggle(TOGGLE_RAW)

    st.subheader(SECTION_SUMMARY)
    st.write(summary if show_raw else summary.get("output_text", ""))

    st.subheader(SECTION_QA)
    for question, answer in answers.items():
        st.markdown(f"**• {question}**")
        st.write(answer if show_raw else _safe_extract_answer(answer))


def _safe_extract_answer(answer):
    return answer.get("result") if isinstance(answer, dict) else str(answer)


def _display_model_info():  # Show which LLM is being used.
    if OPENAI_API_KEY:
        st.info(OPEN_AI_TITLE)
    else:
        st.info(OLLAMA_TITLE)


st.set_page_config(page_title=PAGE_TITLE, layout="wide")  # Set page title
st.title(APP_TITLE)
_display_model_info()

uploaded_file = st.file_uploader("Upload a document", type=FILE_TYPES)

if uploaded_file:
    file_hash, json_path = _get_hash_and_path(uploaded_file)
    progress_text, progress_bar, on_progress = _init_progress_bar()
    regenerate = st.button(BUTTON_REGENERATE)

    if regenerate or not os.path.exists(json_path):
        with st.spinner(MSG_PROCESSING):
            summary, answers = process_document(
                uploaded_file, json_path, on_progress=on_progress
            )
            _clear_progress(progress_text, progress_bar)
    else:
        summary, answers = _load_cached_results(json_path)
        on_progress(100, MSG_DONE)

    _render_output(summary, answers)
