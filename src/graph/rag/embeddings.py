"""Embedding providers for semantic search."""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""


class VoyageEmbeddings(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "voyage-3-lite") -> None:
        import voyageai

        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> list[float]:
        result = self.client.embed([text], model=self.model)
        return result.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self.client.embed(texts, model=self.model)
        return result.embeddings


class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


def get_embedding_provider(
    provider: str, api_key: str, model: str | None = None
) -> EmbeddingProvider:
    if provider == "voyage":
        return VoyageEmbeddings(api_key, model or "voyage-3-lite")
    elif provider == "openai":
        return OpenAIEmbeddings(api_key, model or "text-embedding-3-small")
    raise ValueError(f"Unknown embedding provider: {provider}")


def serialize_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> list[float]:
    num_floats = len(data) // 4
    return list(struct.unpack(f"{num_floats}f", data))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
