from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama # Local model: Mistral
from langchain_community.chat_models import ChatOpenAI

from utils.constants import HUGGINGFACE_MODEL_NAME

# This module loads the appropriate LLM (Language Model) and embedding model based on the presence of an OPENAI_API_KEY in the environment.

load_dotenv(override=True) # Load environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_llm(temperature=0, model_name=None):  # Create and return the right LLM instance
    if OPENAI_API_KEY:

        return ChatOpenAI(
            temperature=temperature, # Creativity level (0-1) where 0 is factual and 1 is creative
            model_name=model_name or "gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY,
        )
    else:

        return ChatOllama(temperature=temperature, model=model_name or "mistral")


def get_embeddings():  # Selects an embedding model to turn documents into vectors
    if OPENAI_API_KEY:
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    else:
        return HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)
