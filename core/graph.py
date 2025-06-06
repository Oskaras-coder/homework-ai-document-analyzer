from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict
from langchain.schema import Document
from nodes.summarizer import summarize_node
from nodes.qa import qa_node

class GraphState(TypedDict):
    docs: List[Document]
    retriever: any
    summary: str
    qa: Dict[str, str]

def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("summarize", summarize_node)
    builder.add_node("qa_node", qa_node)

    builder.set_entry_point("summarize")
    builder.add_edge("summarize", "qa_node")
    builder.set_finish_point("qa_node")

    return builder.compile()