from langchain.chains import RetrievalQA
from core.questions import QUESTIONS
from core.llm import get_llm
from utils.constants import OPENAI_MODEL_NAME


def qa_node(state):
    llm = get_llm(temperature=0, model_name=OPENAI_MODEL_NAME)
    retriever = state["retriever"]
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answers = {q: qa_chain.invoke(q) for q in QUESTIONS}
    return {"summary": state["summary"], "qa": answers}
