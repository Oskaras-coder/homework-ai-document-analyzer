import os
import json
from core.embedder import create_vectorstore
from core.graph import build_graph
from core.loader import load_and_split

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif hasattr(obj, 'page_content'):
        return {
            "page_content": obj.page_content,
            "metadata": obj.metadata
        }
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj
    
def process_document(file, json_path, on_progress=None):
    if on_progress: on_progress(10, "📄 Splitting document...")
    docs = load_and_split(file)

    if on_progress: on_progress(30, "📌 Creating vectorstore...")
    vectorstore = create_vectorstore(docs)
    retriever = vectorstore.as_retriever()

    if on_progress: on_progress(60, "🤖 Running graph inference...")
    graph = build_graph()
    result = graph.invoke({
        "docs": docs,
        "retriever": retriever
    })

    if on_progress: on_progress(90, "💾 Saving result...")
    
    safe_result = make_json_safe(result)
    os.makedirs("results", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(safe_result, f, ensure_ascii=False, indent=4)

    if on_progress: on_progress(100, "✅ Done")
    return result["summary"], result["qa"]
