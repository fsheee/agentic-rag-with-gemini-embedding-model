import pytest

from app import embedding


def test_raises_when_embedding_model_not_configured(monkeypatch):
    monkeypatch.setattr(embedding, "GOOGLE_API_KEY", "google-key")
    monkeypatch.setattr(embedding, "EMBEDDING_MODEL", None)

    with pytest.raises(ValueError, match="EMBEDDING_MODEL is not set"):
        embedding.get_embeddings()


def test_raises_when_google_api_key_not_configured(monkeypatch):
    monkeypatch.setattr(embedding, "GOOGLE_API_KEY", None)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY is not set"):
        embedding.get_embeddings()


def test_passes_configured_model_to_gemini(monkeypatch):
    captured = {}

    monkeypatch.setattr(embedding, "GOOGLE_API_KEY", "google-key")
    monkeypatch.setattr(
        embedding,
        "EMBEDDING_MODEL",
        "models/gemini-embedding-001",
    )
    monkeypatch.setattr(
        embedding,
        "GoogleGenerativeAIEmbeddings",
        lambda **kwargs: captured.update(kwargs) or "embeddings",
    )

    result = embedding.get_embeddings()

    assert result == "embeddings"
    assert captured["model"] == "models/gemini-embedding-001"
    assert captured["google_api_key"] == "google-key"