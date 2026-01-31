"""
Temporal Retrieval

Time-based retrieval strategies for memories.
"""

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from agentmem.core.memory import MemoryItem, MemoryType


class TemporalRetrieval:
    """
    Time-based memory retrieval.

    Supports:
    - Recent memories
    - Memories from specific time periods
    - Temporal patterns
    - Time-weighted relevance
    """

    def __init__(self, db_client=None):
        """
        Initialize temporal retrieval.

        Args:
            db_client: Database client
        """
        self.db = db_client
        self._cache: list[MemoryItem] = []

    async def get_recent(
        self,
        space_id: UUID,
        limit: int = 10,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryItem]:
        """
        Get most recently created memories.

        Args:
            space_id: Memory space
            limit: Maximum results
            memory_type: Filter by type

        Returns:
            List of recent memories
        """
        if self.db:
            query = (
                self.db.table("memories")
                .select("*")
                .eq("space_id", str(space_id))
            )

            if memory_type:
                query = query.eq("memory_type", memory_type.value)

            result = (
                query.order("created_at", desc=True)
                .limit(limit)
                .execute()
            )

            return [MemoryItem.from_dict(row) for row in result.data]

        # In-memory fallback
        memories = [m for m in self._cache if m.space_id == space_id]
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]

        return sorted(memories, key=lambda m: m.created_at, reverse=True)[:limit]

    async def get_recently_accessed(
        self,
        space_id: UUID,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """
        Get most recently accessed memories.

        Args:
            space_id: Memory space
            limit: Maximum results

        Returns:
            List of recently accessed memories
        """
        if self.db:
            result = (
                self.db.table("memories")
                .select("*")
                .eq("space_id", str(space_id))
                .order("accessed_at", desc=True)
                .limit(limit)
                .execute()
            )

            return [MemoryItem.from_dict(row) for row in result.data]

        memories = [m for m in self._cache if m.space_id == space_id]
        return sorted(memories, key=lambda m: m.accessed_at, reverse=True)[:limit]

    async def get_in_timerange(
        self,
        space_id: UUID,
        start_time: datetime,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """
        Get memories created within a time range.

        Args:
            space_id: Memory space
            start_time: Start of range
            end_time: End of range (default: now)
            limit: Maximum results

        Returns:
            List of memories in range
        """
        end_time = end_time or datetime.utcnow()

        if self.db:
            result = (
                self.db.table("memories")
                .select("*")
                .eq("space_id", str(space_id))
                .gte("created_at", start_time.isoformat())
                .lte("created_at", end_time.isoformat())
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )

            return [MemoryItem.from_dict(row) for row in result.data]

        memories = [
            m
            for m in self._cache
            if m.space_id == space_id
            and start_time <= m.created_at <= end_time
        ]
        return sorted(memories, key=lambda m: m.created_at, reverse=True)[:limit]

    async def get_today(
        self,
        space_id: UUID,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """Get memories from today."""
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        return await self.get_in_timerange(space_id, today_start, limit=limit)

    async def get_this_week(
        self,
        space_id: UUID,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """Get memories from this week."""
        now = datetime.utcnow()
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        return await self.get_in_timerange(space_id, week_start, limit=limit)

    async def get_by_hour(
        self,
        space_id: UUID,
        hours_ago: int = 1,
        limit: int = 20,
    ) -> list[MemoryItem]:
        """Get memories from the last N hours."""
        start_time = datetime.utcnow() - timedelta(hours=hours_ago)
        return await self.get_in_timerange(space_id, start_time, limit=limit)

    async def get_by_day(
        self,
        space_id: UUID,
        days_ago: int = 1,
        limit: int = 50,
    ) -> list[MemoryItem]:
        """Get memories from the last N days."""
        start_time = datetime.utcnow() - timedelta(days=days_ago)
        return await self.get_in_timerange(space_id, start_time, limit=limit)

    async def get_temporal_context(
        self,
        space_id: UUID,
        reference_time: datetime,
        window_hours: int = 2,
    ) -> list[MemoryItem]:
        """
        Get memories around a specific time.

        Args:
            space_id: Memory space
            reference_time: Center of time window
            window_hours: Hours before and after

        Returns:
            Memories within the window
        """
        start_time = reference_time - timedelta(hours=window_hours)
        end_time = reference_time + timedelta(hours=window_hours)
        return await self.get_in_timerange(space_id, start_time, end_time)

    def time_weighted_score(
        self,
        memory: MemoryItem,
        reference_time: datetime | None = None,
        decay_hours: float = 24,
    ) -> float:
        """
        Calculate time-weighted relevance score.

        Args:
            memory: Memory to score
            reference_time: Reference point (default: now)
            decay_hours: Hours for score to halve

        Returns:
            Time-weighted score 0-1
        """
        import math

        reference_time = reference_time or datetime.utcnow()
        hours_diff = abs((reference_time - memory.created_at).total_seconds() / 3600)

        # Exponential decay
        decay_rate = math.log(2) / decay_hours
        return math.exp(-decay_rate * hours_diff)

    def rank_by_time_relevance(
        self,
        memories: list[MemoryItem],
        reference_time: datetime | None = None,
        combine_with_importance: bool = True,
    ) -> list[tuple[MemoryItem, float]]:
        """
        Rank memories by temporal relevance.

        Args:
            memories: Memories to rank
            reference_time: Reference time
            combine_with_importance: Also consider importance

        Returns:
            Ranked list of (memory, score) tuples
        """
        scored = []
        for memory in memories:
            time_score = self.time_weighted_score(memory, reference_time)

            if combine_with_importance:
                # Combine time and importance
                final_score = (time_score * 0.6) + (memory.importance * 0.4)
            else:
                final_score = time_score

            scored.append((memory, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
