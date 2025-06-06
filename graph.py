from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict
from langchain.schema import Document
from summarizer import summarize_node
from qa import qa_node

class GraphState(TypedDict):
    docs: List[Document]
    retriever: any
    summary: str
    qa: Dict[str, str]

# ✅ Step 2: Build the LangGraph with the schema
def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("summarize", summarize_node)
    builder.add_node("qa_node", qa_node)

    builder.set_entry_point("summarize")
    builder.add_edge("summarize", "qa_node")
    builder.set_finish_point("qa_node")

    return builder.compile()