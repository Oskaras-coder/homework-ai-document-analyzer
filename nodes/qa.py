from langchain.chains import RetrievalQA
from core.questions import QUESTIONS
from core.llm import get_llm
from utils.constants import OPENAI_MODEL_NAME


def qa_node(state):
    llm = get_llm(temperature=0, model_name=OPENAI_MODEL_NAME)
    retriever = state["retriever"] # Retrieve the FAISS retriever from the shared graph state
    #  Build the RetrievalQA chain:
    #    - It embeds the query
    #    - Finds the top-k relevant chunks via `retriever`
    #    - Feeds them to the LLM for answer generation
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Execute the chain for each question, collecting the plain-text answers 
    answers = {               
        q: qa_chain.invoke(q) # invoke() sends the question + retrieved context to the LLM
        for q in QUESTIONS
        } 
    return {"summary": state["summary"], "qa": answers}
