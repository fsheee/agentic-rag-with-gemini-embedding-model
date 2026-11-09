
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from .config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    MODEL_NAME,
)


def get_llm():
    """Initialize the configured chat model."""

    if GOOGLE_API_KEY:
        if not GEMINI_MODEL:
            raise ValueError(
                "GEMINI_MODEL is not set in your .env file."
            )

        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )

    if GROQ_API_KEY:
        if not MODEL_NAME:
            raise ValueError(
                "MODEL_NAME is not set in your .env file."
            )

        return ChatGroq(
            model=MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )

    raise ValueError(
        "No LLM API key configured. Set GOOGLE_API_KEY or GROQ_API_KEY."
    )
