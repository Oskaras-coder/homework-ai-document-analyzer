from langchain_community.vectorstores import FAISS

from llm import get_embeddings

def create_vectorstore(documents):
    embedding = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embedding)
    return vectorstore