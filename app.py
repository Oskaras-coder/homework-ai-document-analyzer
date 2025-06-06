import json
import os
import streamlit as st
from main import process_document
from utils.utils import get_file_hash

st.set_page_config(page_title="AI Document Analyzer", layout="wide")
st.title("📄 AI-Powered Document Analyzer (LangChain + LangGraph + RAG)")

uploaded_file = st.file_uploader("Upload a document (PDF only)", type=["pdf"])

if uploaded_file:
    file_hash = get_file_hash(uploaded_file)
    json_path = f"results/{file_hash}.json"
    progress_text = st.empty()
    progress_bar = st.progress(0)

    regenerate = st.button("♻️ Regenerate Analysis")
    def on_progress(pct, message):
        progress_text.text(message)
        progress_bar.progress(pct)

    if regenerate or not os.path.exists(json_path):
        with st.spinner("Processing... This might take a minute ⏳"):
            summary, answers = process_document(uploaded_file, json_path, on_progress=on_progress)
            progress_bar.empty()
            progress_text.empty()
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
            summary = cached["summary"]
            answers = cached["qa"]
        on_progress(100, "✅ Loaded from cache")
        
    # 🔘 Toggle for raw/clean view
    show_raw = st.toggle("🛠 Show raw output")

    st.subheader("🔍 Summary")
    if show_raw:
        st.write(summary)  # could be string or dict, safe
    else:
        st.write(summary["output_text"])  # summary is a plain string now

    st.subheader("🧠 Answers to Predefined Questions")
    for question, answer in answers.items():
        if show_raw:
            st.markdown(f"**• {question}**")
            st.write(answer)  # full dict (e.g. {"query":..., "result":...})
        else:
            # Flexible: if answer is a dict, show "result"; otherwise show the raw string
            result = answer.get("result") if isinstance(answer, dict) else str(answer)
            st.markdown(f"**• {question}**")
            st.write(result)
        