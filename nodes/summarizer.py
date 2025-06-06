from langchain.chains.summarize import load_summarize_chain
from core.llm import get_llm
from core.text_splitter import split_documents

def summarize_node(state):
    docs = state["docs"]
    split_docs = split_documents(docs)
    llm = get_llm(temperature=0.3)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.invoke(split_docs)
    return {"summary": summary, "retriever": state["retriever"]}