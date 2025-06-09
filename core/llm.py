from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOpenAI

from utils.constants import HUGGINGFACE_MODEL_NAME

load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_llm(temperature=0, model_name=None):
    if OPENAI_API_KEY:

        return ChatOpenAI(
            temperature=temperature,
            model_name=model_name or "gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY,
        )
    else:

        return ChatOllama(temperature=temperature, model=model_name or "mistral")


def get_embeddings():
    if OPENAI_API_KEY:
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    else:
        return HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)
