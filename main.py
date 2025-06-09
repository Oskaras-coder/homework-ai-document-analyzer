import os
import json
from core.loader import load_and_split
from core.embedder import create_vectorstore
from core.graph import build_graph
from utils.serialization import make_json_safe
from utils.constants import (
    MSG_SPLITTING,
    MSG_CREATING_VECTORSTORE,
    MSG_RUNNING_GRAPH,
    MSG_SAVING_RESULT,
    MSG_DONE,
)


def process_document(file, json_path, on_progress=None):
    docs = _prepare_documents(file, on_progress)
    retriever = _prepare_retriever(docs, on_progress)
    result = _run_pipeline(docs, retriever, on_progress)
    _save_result(result, json_path, on_progress)
    return result["summary"], result["qa"]


def _prepare_documents(file, on_progress):
    if on_progress:
        on_progress(10, MSG_SPLITTING)
    return load_and_split(file)


def _prepare_retriever(docs, on_progress):
    if on_progress:
        on_progress(30, MSG_CREATING_VECTORSTORE)
    vectorstore = create_vectorstore(docs)
    return vectorstore.as_retriever()


def _run_pipeline(docs, retriever, on_progress):
    if on_progress:
        on_progress(60, MSG_RUNNING_GRAPH)
    graph = build_graph()
    return graph.invoke({"docs": docs, "retriever": retriever})


def _save_result(result, json_path, on_progress):
    if on_progress:
        on_progress(90, MSG_SAVING_RESULT)
    safe_result = make_json_safe(result)
    os.makedirs("results", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe_result, f, ensure_ascii=False, indent=4)
    if on_progress:
        on_progress(100, MSG_DONE)
