"""
Semantic Search

Vector-based semantic search for memories.
"""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from agentmem.core.memory import MemoryItem, MemoryType


@dataclass
class SearchResult:
    """Result from semantic search."""

    memory: MemoryItem
    score: float
    explanation: str | None = None


class EmbeddingProvider:
    """
    Provider for generating text embeddings.

    Supports multiple backends:
    - OpenAI ada-002
    - Local sentence-transformers
    - Kimi (cost-effective)
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize embedding provider.

        Args:
            provider: Provider name (openai, local, kimi)
            model: Model name/ID
            api_key: API key for remote providers
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key

        if provider == "openai":
            self.model = model or "text-embedding-ada-002"
            self._dimension = 1536
        elif provider == "local":
            self.model = model or "all-MiniLM-L6-v2"
            self._dimension = 384
        else:
            self._dimension = 1536

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.provider == "openai":
            return await self._openai_embed(text)
        elif self.provider == "local":
            return self._local_embed(text)
        else:
            # Fallback to simple hash-based embedding
            return self._hash_embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of embedding vectors
        """
        if self.provider == "openai":
            return await self._openai_embed_batch(texts)
        else:
            return [await self.embed(text) for text in texts]

    async def _openai_embed(self, text: str) -> list[float]:
        """OpenAI embedding."""
        import httpx
        import os

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "input": text,
                    "model": self.model,
                },
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    async def _openai_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch OpenAI embedding."""
        import httpx
        import os

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "input": texts,
                    "model": self.model,
                },
            )
            response.raise_for_status()
            data = response.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def _local_embed(self, text: str) -> list[float]:
        """Local sentence-transformers embedding."""
        try:
            from sentence_transformers import SentenceTransformer

            if not hasattr(self, "_local_model"):
                self._local_model = SentenceTransformer(self.model)

            return self._local_model.encode(text).tolist()
        except ImportError:
            return self._hash_embed(text)

    def _hash_embed(self, text: str) -> list[float]:
        """Simple hash-based embedding for testing."""
        import hashlib

        # Create a deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []

        for i in range(0, min(len(hash_bytes), self._dimension), 1):
            embedding.append((hash_bytes[i % len(hash_bytes)] - 128) / 128.0)

        # Pad or truncate to dimension
        while len(embedding) < self._dimension:
            embedding.append(0.0)

        return embedding[: self._dimension]


class SemanticSearch:
    """
    Semantic search engine for memories.

    Uses vector embeddings to find semantically similar memories.
    Supports filtering by type, importance, and time.
    """

    def __init__(
        self,
        db_client=None,
        embedding_provider: EmbeddingProvider | None = None,
        cache=None,
    ):
        """
        Initialize semantic search.

        Args:
            db_client: Database client
            embedding_provider: Embedding provider
            cache: Cache for search results
        """
        self.db = db_client
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.cache = cache

    async def search(
        self,
        query: str,
        space_id: UUID,
        limit: int = 10,
        min_score: float = 0.5,
        memory_type: MemoryType | None = None,
        min_importance: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search for memories semantically similar to query.

        Args:
            query: Search query
            space_id: Memory space to search
            limit: Maximum results
            min_score: Minimum similarity score
            memory_type: Filter by type
            min_importance: Minimum importance threshold

        Returns:
            List of SearchResults
        """
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed(query)

        # Check cache
        if self.cache:
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            cached = await self.cache.get_search_results(query_hash, str(space_id))
            if cached:
                return [
                    SearchResult(
                        memory=MemoryItem.from_dict(r["memory"]),
                        score=r["score"],
                    )
                    for r in cached
                ]

        # Search database
        results = []

        if self.db:
            # Use pgvector similarity search
            result = self.db.rpc(
                "match_memories",
                {
                    "query_embedding": query_embedding,
                    "match_count": limit * 2,  # Get more for filtering
                    "match_threshold": min_score,
                    "p_space_id": str(space_id),
                },
            ).execute()

            for row in result.data:
                memory = MemoryItem.from_dict(row)

                # Apply filters
                if memory_type and memory.memory_type != memory_type:
                    continue
                if memory.importance < min_importance:
                    continue

                results.append(
                    SearchResult(
                        memory=memory,
                        score=row.get("similarity", 0.0),
                    )
                )

                if len(results) >= limit:
                    break

        # Cache results
        if self.cache and results:
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            await self.cache.set_search_results(
                query_hash,
                str(space_id),
                [
                    {"memory": r.memory.to_dict(), "score": r.score}
                    for r in results
                ],
            )

        return results

    async def find_similar(
        self,
        memory: MemoryItem,
        limit: int = 5,
        min_score: float = 0.7,
    ) -> list[SearchResult]:
        """
        Find memories similar to a given memory.

        Args:
            memory: Reference memory
            limit: Maximum results
            min_score: Minimum similarity

        Returns:
            List of similar memories
        """
        if memory.embedding:
            query_embedding = memory.embedding
        else:
            query_embedding = await self.embedding_provider.embed(memory.content)

        results = await self.search(
            memory.content,
            memory.space_id,
            limit=limit + 1,  # +1 to exclude self
            min_score=min_score,
        )

        # Exclude the memory itself
        return [r for r in results if r.memory.id != memory.id][:limit]

    async def rank_by_relevance(
        self,
        query: str,
        memories: list[MemoryItem],
    ) -> list[tuple[MemoryItem, float]]:
        """
        Rank a list of memories by relevance to query.

        Args:
            query: Query to rank against
            memories: Memories to rank

        Returns:
            List of (memory, score) tuples, sorted by score
        """
        if not memories:
            return []

        query_embedding = await self.embedding_provider.embed(query)

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            import math
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        ranked = []
        for memory in memories:
            if memory.embedding:
                score = cosine_similarity(query_embedding, memory.embedding)
            else:
                # Generate embedding for unembedded memories
                mem_embedding = await self.embedding_provider.embed(memory.content)
                score = cosine_similarity(query_embedding, mem_embedding)

            ranked.append((memory, score))

        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
