from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_llm(temperature: float = 0.3, model_name: str = "gpt-3.5-turbo"):
    return ChatOpenAI(
        temperature=temperature,
        model_name=model_name,
        openai_api_key=OPENAI_API_KEY
    )

def get_embeddings():
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY)