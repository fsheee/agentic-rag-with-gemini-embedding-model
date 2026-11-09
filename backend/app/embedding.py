
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initialize and return the Gemini embedding model.
    """

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Add it to your .env file."
        )

    if not EMBEDDING_MODEL:
        raise ValueError(
            "EMBEDDING_MODEL is not set. Add it to your .env file."
        )

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
