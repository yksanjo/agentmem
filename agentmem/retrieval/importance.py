"""
Importance Scoring

Calculates and manages importance scores for memories.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from agentmem.core.memory import MemoryItem


@dataclass
class ImportanceFactors:
    """Factors used to calculate importance."""

    recency_weight: float = 0.3
    frequency_weight: float = 0.2
    explicit_weight: float = 0.3
    semantic_weight: float = 0.2


class ImportanceScorer:
    """
    Calculates importance scores for memories.

    Importance is based on multiple factors:
    - Recency: How recently was it accessed?
    - Frequency: How often is it accessed?
    - Explicit: User-assigned importance
    - Semantic: Relevance to common queries

    Higher importance = longer retention, priority in search.
    """

    def __init__(
        self,
        factors: ImportanceFactors | None = None,
        db_client=None,
    ):
        """
        Initialize importance scorer.

        Args:
            factors: Weight factors for scoring
            db_client: Database client
        """
        self.factors = factors or ImportanceFactors()
        self.db = db_client

    def calculate_score(
        self,
        memory: MemoryItem,
        context: dict[str, Any] | None = None,
    ) -> float:
        """
        Calculate overall importance score for a memory.

        Args:
            memory: Memory to score
            context: Optional context (e.g., current query)

        Returns:
            Importance score 0-1
        """
        scores = []

        # Recency score
        recency = self._recency_score(memory)
        scores.append((recency, self.factors.recency_weight))

        # Frequency score
        frequency = self._frequency_score(memory)
        scores.append((frequency, self.factors.frequency_weight))

        # Explicit importance
        explicit = memory.importance
        scores.append((explicit, self.factors.explicit_weight))

        # Semantic relevance (if context provided)
        if context and "query" in context:
            semantic = self._semantic_score(memory, context["query"])
            scores.append((semantic, self.factors.semantic_weight))
        else:
            # Use explicit as fallback
            scores.append((explicit, self.factors.semantic_weight))

        # Weighted average
        total_weight = sum(w for _, w in scores)
        weighted_sum = sum(s * w for s, w in scores)

        return min(1.0, max(0.0, weighted_sum / total_weight))

    def _recency_score(self, memory: MemoryItem) -> float:
        """Calculate recency score based on last access."""
        now = datetime.utcnow()
        hours_since_access = (now - memory.accessed_at).total_seconds() / 3600

        # Exponential decay
        # Score = 1.0 at 0 hours, ~0.5 at 24 hours, ~0.1 at 72 hours
        decay_rate = 0.03
        return max(0.0, min(1.0, 1.0 - (hours_since_access * decay_rate)))

    def _frequency_score(self, memory: MemoryItem) -> float:
        """Calculate frequency score based on access count."""
        # Logarithmic scaling
        # 0 accesses = 0.0, 1 = 0.3, 10 = 0.7, 100 = 1.0
        import math

        if memory.access_count == 0:
            return 0.0

        return min(1.0, math.log10(memory.access_count + 1) / 2)

    def _semantic_score(self, memory: MemoryItem, query: str) -> float:
        """Calculate semantic relevance to query."""
        # Simple keyword matching for now
        # In production, use embedding similarity
        query_lower = query.lower()
        content_lower = memory.content.lower()

        # Count matching words
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = len(query_words & content_words)

        if not query_words:
            return 0.0

        return min(1.0, overlap / len(query_words))

    async def update_importance(
        self,
        memory_id: str,
        delta: float,
        reason: str | None = None,
    ) -> float:
        """
        Update a memory's importance by delta.

        Args:
            memory_id: Memory to update
            delta: Change in importance (-1 to 1)
            reason: Reason for change

        Returns:
            New importance value
        """
        if self.db:
            # Fetch current importance
            result = (
                self.db.table("memories")
                .select("importance")
                .eq("id", memory_id)
                .execute()
            )

            if not result.data:
                return 0.0

            current = result.data[0]["importance"]
            new_importance = max(0.0, min(1.0, current + delta))

            # Update
            self.db.table("memories").update(
                {"importance": new_importance}
            ).eq("id", memory_id).execute()

            return new_importance

        return 0.5 + delta  # Fallback

    async def decay_all(
        self,
        space_id: str,
        decay_rate: float = 0.01,
    ) -> int:
        """
        Apply importance decay to all memories in a space.

        Args:
            space_id: Space to decay
            decay_rate: Rate of decay

        Returns:
            Number of memories updated
        """
        if not self.db:
            return 0

        # Fetch all memories
        result = (
            self.db.table("memories")
            .select("id, importance")
            .eq("space_id", space_id)
            .execute()
        )

        updated = 0
        for row in result.data:
            current = row["importance"]
            new_importance = max(0.0, current - decay_rate)

            if new_importance != current:
                self.db.table("memories").update(
                    {"importance": new_importance}
                ).eq("id", row["id"]).execute()
                updated += 1

        return updated

    async def boost_accessed(
        self,
        memory_id: str,
        boost: float = 0.05,
    ) -> float:
        """
        Boost importance when memory is accessed.

        Args:
            memory_id: Memory that was accessed
            boost: Boost amount

        Returns:
            New importance
        """
        return await self.update_importance(
            memory_id,
            boost,
            reason="access_boost",
        )

    def rank_by_importance(
        self,
        memories: list[MemoryItem],
        context: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """
        Rank memories by calculated importance.

        Args:
            memories: Memories to rank
            context: Optional context for scoring

        Returns:
            List of (memory, score) tuples, sorted by score
        """
        scored = [
            (memory, self.calculate_score(memory, context))
            for memory in memories
        ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
