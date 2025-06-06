from langchain.chains import RetrievalQA
from questions import QUESTIONS
from llm import get_llm  # ✅ import the shared LLM

def qa_node(state):
    llm = get_llm(temperature=0)  # lower temperature for factual QA
    retriever = state["retriever"]
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answers = {q: qa_chain.invoke(q) for q in QUESTIONS}
    return {"summary": state["summary"], "qa": answers}