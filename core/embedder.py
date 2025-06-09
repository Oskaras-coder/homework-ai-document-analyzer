from langchain_community.vectorstores import FAISS

from core.llm import get_embeddings


def create_vectorstore(documents):
    embedding = get_embeddings()
    return FAISS.from_documents(documents, embedding)
