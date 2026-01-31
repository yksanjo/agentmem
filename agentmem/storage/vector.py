"""
Vector Store

Vector storage with pgvector or Qdrant for semantic search.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass
class VectorSearchResult:
    """Result from vector search."""

    id: str
    score: float
    metadata: dict[str, Any]
    content: str | None = None


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> bool:
        """Insert or update a vector."""
        pass

    @abstractmethod
    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete a vector."""
        pass


class PgVectorStore(VectorStore):
    """
    PostgreSQL vector store using pgvector extension.

    Requires:
    - PostgreSQL with pgvector extension
    - Supabase (includes pgvector by default)
    """

    def __init__(
        self,
        db_client,
        table_name: str = "memories",
        embedding_dimension: int = 1536,
    ):
        """
        Initialize pgvector store.

        Args:
            db_client: Supabase client
            table_name: Table containing vectors
            embedding_dimension: Vector dimension (1536 for ada-002)
        """
        self.db = db_client
        self.table_name = table_name
        self.dimension = embedding_dimension

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> bool:
        """Insert or update a vector."""
        try:
            self.db.table(self.table_name).upsert(
                {
                    "id": id,
                    "embedding": embedding,
                    "metadata": metadata,
                }
            ).execute()
            return True
        except Exception:
            return False

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search for similar vectors using cosine similarity.

        Requires a match_memories function in Supabase:
        ```sql
        CREATE FUNCTION match_memories (
            query_embedding vector(1536),
            match_count int,
            match_threshold float,
            p_space_id uuid
        ) RETURNS TABLE (
            id uuid,
            content text,
            metadata jsonb,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                m.id,
                m.content,
                m.metadata,
                1 - (m.embedding <=> query_embedding) as similarity
            FROM memories m
            WHERE m.space_id = p_space_id
                AND 1 - (m.embedding <=> query_embedding) > match_threshold
            ORDER BY m.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$;
        ```
        """
        space_id = filter.get("space_id") if filter else None

        result = self.db.rpc(
            "match_memories",
            {
                "query_embedding": embedding,
                "match_count": limit,
                "match_threshold": 0.5,
                "p_space_id": space_id,
            },
        ).execute()

        return [
            VectorSearchResult(
                id=str(row["id"]),
                score=row.get("similarity", 0.0),
                metadata=row.get("metadata", {}),
                content=row.get("content"),
            )
            for row in result.data
        ]

    async def delete(self, id: str) -> bool:
        """Delete a vector."""
        try:
            self.db.table(self.table_name).delete().eq("id", id).execute()
            return True
        except Exception:
            return False


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store for development/testing.

    Uses numpy for similarity calculations if available,
    falls back to pure Python otherwise.
    """

    def __init__(self):
        """Initialize in-memory store."""
        self._vectors: dict[str, tuple[list[float], dict[str, Any]]] = {}

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> bool:
        """Insert or update a vector."""
        self._vectors[id] = (embedding, metadata)
        return True

    async def search(
        self,
        embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar vectors."""
        import math

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        results = []
        for id, (vec, metadata) in self._vectors.items():
            # Apply filter
            if filter:
                match = True
                for key, value in filter.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            score = cosine_similarity(embedding, vec)
            results.append(
                VectorSearchResult(
                    id=id,
                    score=score,
                    metadata=metadata,
                )
            )

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def delete(self, id: str) -> bool:
        """Delete a vector."""
        if id in self._vectors:
            del self._vectors[id]
            return True
        return False

    @property
    def count(self) -> int:
        """Get vector count."""
        return len(self._vectors)
