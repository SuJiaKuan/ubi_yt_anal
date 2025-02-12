import os

from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv


def create_openai_llm(
    model_name: str = "gpt-4o-mini-2024-07-18", temperature: float = 0.7
) -> ChatOpenAI:
    load_dotenv()

    return ChatOpenAI(
        model=model_name, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY")
    )
