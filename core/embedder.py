from langchain_community.vectorstores import FAISS

from core.llm import get_embeddings


def create_vectorstore(
    documents,
):  # Converts all documents into vectors and stores them in a FAISS index
    embedding = get_embeddings()
    return FAISS.from_documents(documents, embedding)


# Without the vectorstore, LLM won’t be able to find context to answer questions or summarize sections accurately.
