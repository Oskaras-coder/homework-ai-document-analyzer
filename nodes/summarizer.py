from langchain.chains.summarize import load_summarize_chain
from core.llm import get_llm
from utils.constants import OPENAI_MODEL_NAME


def summarize_node(state):
    docs = state["docs"]
    llm = get_llm(temperature=0, model_name=OPENAI_MODEL_NAME)
    chain = load_summarize_chain(
        llm, chain_type="map_reduce"
    )  # Map-reduce pipeline: summarize each chunk, then combine
    summary = chain.invoke(
        docs
    )  # Run the summarization chain on all chunks at once: Map + Reduce
    return {"summary": summary, "retriever": state["retriever"]}
