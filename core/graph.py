from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict
from langchain.schema import Document
from nodes.summarizer import summarize_node
from nodes.qa import qa_node


class GraphState(TypedDict):
    docs: List[Document]     # List of LangChain Documents (split & processed PDF chunks)
    retriever: any           # Retriever object created from vector store: FAISS
    summary: str             # Summary output (returned by summarize_node)
    qa: Dict[str, str]       # Dictionary of Q&A results (returned by qa_node)

# Controls the sequence of nodes (steps) for document processing
def build_graph():
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("summarize", summarize_node)
    builder.add_node("qa_node", qa_node)

    # Add steps
    builder.set_entry_point("summarize")
    builder.add_edge("summarize", "qa_node")
    builder.set_finish_point("qa_node")

    return builder.compile()

# LangGraph enables state passing between steps, so each node only needs to focus on its own logic.