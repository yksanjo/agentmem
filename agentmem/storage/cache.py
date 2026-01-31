"""
Memory Cache

Redis-based cache layer for fast memory access.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any
import json


@dataclass
class CacheConfig:
    """Cache configuration."""

    default_ttl: int = 3600  # 1 hour
    max_size: int = 10000
    prefix: str = "agentmem"


class MemoryCache:
    """
    Redis cache layer for memories.

    Provides fast access to frequently used memories
    with automatic TTL-based expiration.
    """

    def __init__(
        self,
        redis_client=None,
        config: CacheConfig | None = None,
    ):
        """
        Initialize memory cache.

        Args:
            redis_client: Redis client
            config: Cache configuration
        """
        self.redis = redis_client
        self.config = config or CacheConfig()
        self._local_cache: dict[str, tuple[Any, float]] = {}  # Fallback

    def _key(self, *parts: str) -> str:
        """Generate cache key."""
        return f"{self.config.prefix}:{':'.join(parts)}"

    async def get(self, key: str) -> Any | None:
        """Get a cached value."""
        cache_key = self._key(key)

        if self.redis:
            value = self.redis.get(cache_key)
            if value:
                return json.loads(value)
            return None

        # Fallback to local cache
        import time
        if cache_key in self._local_cache:
            value, expires_at = self._local_cache[cache_key]
            if time.time() < expires_at:
                return value
            del self._local_cache[cache_key]
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set a cached value."""
        cache_key = self._key(key)
        ttl = ttl or self.config.default_ttl

        if self.redis:
            self.redis.setex(
                cache_key,
                ttl,
                json.dumps(value, default=str),
            )
            return True

        # Fallback to local cache
        import time
        self._local_cache[cache_key] = (value, time.time() + ttl)
        return True

    async def delete(self, key: str) -> bool:
        """Delete a cached value."""
        cache_key = self._key(key)

        if self.redis:
            return bool(self.redis.delete(cache_key))

        if cache_key in self._local_cache:
            del self._local_cache[cache_key]
            return True
        return False

    async def get_memory(self, memory_id: str) -> dict | None:
        """Get a cached memory."""
        return await self.get(f"memory:{memory_id}")

    async def set_memory(
        self,
        memory_id: str,
        memory_data: dict,
        ttl: int | None = None,
    ) -> bool:
        """Cache a memory."""
        return await self.set(f"memory:{memory_id}", memory_data, ttl)

    async def invalidate_memory(self, memory_id: str) -> bool:
        """Invalidate a cached memory."""
        return await self.delete(f"memory:{memory_id}")

    async def get_search_results(
        self,
        query_hash: str,
        space_id: str,
    ) -> list[dict] | None:
        """Get cached search results."""
        return await self.get(f"search:{space_id}:{query_hash}")

    async def set_search_results(
        self,
        query_hash: str,
        space_id: str,
        results: list[dict],
        ttl: int = 300,  # 5 minutes for search results
    ) -> bool:
        """Cache search results."""
        return await self.set(f"search:{space_id}:{query_hash}", results, ttl)

    async def get_space_stats(self, space_id: str) -> dict | None:
        """Get cached space statistics."""
        return await self.get(f"stats:{space_id}")

    async def set_space_stats(
        self,
        space_id: str,
        stats: dict,
        ttl: int = 60,  # 1 minute for stats
    ) -> bool:
        """Cache space statistics."""
        return await self.set(f"stats:{space_id}", stats, ttl)

    async def clear_space(self, space_id: str) -> int:
        """Clear all cache entries for a space."""
        if self.redis:
            pattern = self._key(f"*:{space_id}:*")
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
            return 0

        # Fallback
        prefix = f"{space_id}:"
        to_delete = [k for k in self._local_cache if prefix in k]
        for key in to_delete:
            del self._local_cache[key]
        return len(to_delete)

    async def get_working_memory(self, session_id: str) -> list[dict] | None:
        """Get working memory for a session."""
        return await self.get(f"working:{session_id}")

    async def set_working_memory(
        self,
        session_id: str,
        memories: list[dict],
        ttl: int = 3600,
    ) -> bool:
        """Set working memory for a session."""
        return await self.set(f"working:{session_id}", memories, ttl)

    async def append_working_memory(
        self,
        session_id: str,
        memory: dict,
        max_items: int = 50,
    ) -> bool:
        """Append to working memory."""
        key = f"working:{session_id}"
        current = await self.get(key) or []
        current.append(memory)

        # Trim to max size
        if len(current) > max_items:
            current = current[-max_items:]

        return await self.set(key, current)
