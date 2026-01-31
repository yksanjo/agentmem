"""
Long-Term Memory

Persistent semantic memory with vector embeddings for retrieval.
Optimized for large-scale storage and semantic search.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from agentmem.core.memory import MemoryItem, MemoryType


@dataclass
class LongTermConfig:
    """Configuration for long-term memory."""

    # Consolidation settings
    consolidation_threshold: int = 100  # Items before consolidation
    consolidation_similarity: float = 0.9  # Similarity for merging

    # Forgetting curve settings
    enable_forgetting: bool = True
    base_retention_days: int = 365
    importance_multiplier: float = 2.0  # High importance = longer retention

    # Search settings
    default_top_k: int = 10
    min_similarity: float = 0.5


class LongTermMemory:
    """
    Long-term memory storage with semantic capabilities.

    Uses vector embeddings for semantic search and retrieval.
    Implements forgetting curves based on importance and access patterns.

    Features:
    - Semantic search via embeddings
    - Importance-weighted retention
    - Automatic consolidation of similar memories
    - Efficient vector storage with pgvector
    """

    def __init__(
        self,
        space_id: UUID,
        db_client=None,
        embedding_provider=None,
        config: LongTermConfig | None = None,
    ):
        """
        Initialize long-term memory.

        Args:
            space_id: Memory space identifier
            db_client: Database client (Supabase with pgvector)
            embedding_provider: Provider for embeddings
            config: Configuration options
        """
        self.space_id = space_id
        self.db = db_client
        self.embedding_provider = embedding_provider
        self.config = config or LongTermConfig()
        self._cache: dict[UUID, MemoryItem] = {}

    async def store(
        self,
        content: str,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """
        Store a new long-term memory.

        Args:
            content: Memory content
            importance: Importance score 0-1
            metadata: Additional metadata

        Returns:
            Created MemoryItem
        """
        # Generate embedding
        embedding = None
        if self.embedding_provider:
            embedding = await self.embedding_provider.embed(content)

        memory = MemoryItem(
            id=uuid4(),
            space_id=self.space_id,
            content=content,
            embedding=embedding,
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            metadata=metadata or {},
        )

        if self.db:
            self.db.table("memories").insert(
                {
                    "id": str(memory.id),
                    "space_id": str(memory.space_id),
                    "content": memory.content,
                    "embedding": memory.embedding,
                    "memory_type": memory.memory_type.value,
                    "importance": memory.importance,
                    "metadata": memory.metadata,
                }
            ).execute()

        self._cache[memory.id] = memory
        return memory

    async def search(
        self,
        query: str,
        limit: int | None = None,
        min_similarity: float | None = None,
        memory_type: MemoryType | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """
        Semantic search for memories.

        Args:
            query: Search query
            limit: Maximum results
            min_similarity: Minimum similarity threshold
            memory_type: Filter by type

        Returns:
            List of (memory, similarity_score) tuples
        """
        limit = limit or self.config.default_top_k
        min_similarity = min_similarity or self.config.min_similarity

        if not self.embedding_provider:
            # Fallback to text search
            return await self._text_search(query, limit)

        # Generate query embedding
        query_embedding = await self.embedding_provider.embed(query)

        if self.db:
            # Use pgvector for similarity search
            # This requires the pgvector extension and a vector index
            result = self.db.rpc(
                "match_memories",
                {
                    "query_embedding": query_embedding,
                    "match_count": limit,
                    "match_threshold": min_similarity,
                    "p_space_id": str(self.space_id),
                },
            ).execute()

            memories = []
            for row in result.data:
                memory = MemoryItem.from_dict(row)
                similarity = row.get("similarity", 0.0)
                memories.append((memory, similarity))

            return memories

        # In-memory similarity search
        return self._cosine_search(query_embedding, limit, min_similarity)

    async def _text_search(
        self,
        query: str,
        limit: int,
    ) -> list[tuple[MemoryItem, float]]:
        """Fallback text-based search."""
        query_lower = query.lower()
        results = []

        if self.db:
            result = (
                self.db.table("memories")
                .select("*")
                .eq("space_id", str(self.space_id))
                .ilike("content", f"%{query}%")
                .limit(limit)
                .execute()
            )

            for row in result.data:
                memory = MemoryItem.from_dict(row)
                # Simple relevance score based on occurrence
                score = query_lower.count(memory.content.lower()) * 0.1
                results.append((memory, min(score, 1.0)))
        else:
            for memory in self._cache.values():
                if query_lower in memory.content.lower():
                    results.append((memory, 0.5))

        return results[:limit]

    def _cosine_search(
        self,
        query_embedding: list[float],
        limit: int,
        min_similarity: float,
    ) -> list[tuple[MemoryItem, float]]:
        """In-memory cosine similarity search."""
        import math

        def cosine_similarity(a: list[float], b: list[float]) -> float:
            if not a or not b:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        results = []
        for memory in self._cache.values():
            if memory.embedding:
                similarity = cosine_similarity(query_embedding, memory.embedding)
                if similarity >= min_similarity:
                    results.append((memory, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def consolidate(self) -> int:
        """
        Consolidate similar memories to reduce redundancy.

        Returns:
            Number of memories merged
        """
        if not self.embedding_provider:
            return 0

        memories = list(self._cache.values())
        if len(memories) < self.config.consolidation_threshold:
            return 0

        merged_count = 0
        to_delete = set()

        for i, mem1 in enumerate(memories):
            if mem1.id in to_delete:
                continue

            for mem2 in memories[i + 1 :]:
                if mem2.id in to_delete:
                    continue

                if mem1.embedding and mem2.embedding:
                    # Check similarity
                    similarity = self._cosine_similarity(mem1.embedding, mem2.embedding)

                    if similarity >= self.config.consolidation_similarity:
                        # Merge mem2 into mem1
                        merged_content = f"{mem1.content}\n\nRelated: {mem2.content}"
                        mem1.content = merged_content
                        mem1.importance = max(mem1.importance, mem2.importance)
                        mem1.embedding = await self.embedding_provider.embed(
                            merged_content
                        )

                        to_delete.add(mem2.id)
                        merged_count += 1

        # Delete merged memories
        for memory_id in to_delete:
            del self._cache[memory_id]
            if self.db:
                self.db.table("memories").delete().eq(
                    "id", str(memory_id)
                ).execute()

        return merged_count

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def apply_forgetting(self) -> int:
        """
        Apply forgetting curve to remove old, low-importance memories.

        Returns:
            Number of memories forgotten
        """
        if not self.config.enable_forgetting:
            return 0

        now = datetime.utcnow()
        forgotten = 0

        for memory in list(self._cache.values()):
            # Calculate retention based on importance
            retention_days = self.config.base_retention_days * (
                1 + memory.importance * self.config.importance_multiplier
            )

            # Adjust for access patterns
            days_since_access = (now - memory.accessed_at).days
            days_since_creation = (now - memory.created_at).days

            # More accesses = longer retention
            access_bonus = min(memory.access_count * 10, 365)
            effective_retention = retention_days + access_bonus

            if days_since_access > effective_retention:
                del self._cache[memory.id]
                if self.db:
                    self.db.table("memories").delete().eq(
                        "id", str(memory.id)
                    ).execute()
                forgotten += 1

        return forgotten

    async def get_important(self, limit: int = 10) -> list[MemoryItem]:
        """Get most important memories."""
        if self.db:
            result = (
                self.db.table("memories")
                .select("*")
                .eq("space_id", str(self.space_id))
                .order("importance", desc=True)
                .limit(limit)
                .execute()
            )
            return [MemoryItem.from_dict(row) for row in result.data]

        memories = sorted(
            self._cache.values(),
            key=lambda m: m.importance,
            reverse=True,
        )
        return memories[:limit]

    @property
    def count(self) -> int:
        """Get total memory count."""
        if self.db:
            result = (
                self.db.table("memories")
                .select("id", count="exact")
                .eq("space_id", str(self.space_id))
                .execute()
            )
            return result.count or 0
        return len(self._cache)
