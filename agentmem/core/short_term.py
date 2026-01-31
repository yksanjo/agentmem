"""
Short-Term Memory

Session-based memory for current context and recent interactions.
Optimized for fast access and limited retention.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4


@dataclass
class ShortTermItem:
    """A short-term memory item with TTL."""

    id: UUID
    content: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class ShortTermMemory:
    """
    Short-term memory storage.

    Provides fast in-memory storage for current session context.
    Items expire after a configurable TTL.

    Features:
    - LRU eviction when capacity is reached
    - Automatic TTL-based expiration
    - Fast O(1) access
    - Session isolation
    """

    DEFAULT_TTL_SECONDS = 3600  # 1 hour
    DEFAULT_MAX_ITEMS = 100

    def __init__(
        self,
        session_id: str | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        redis_client=None,
    ):
        """
        Initialize short-term memory.

        Args:
            session_id: Unique session identifier
            max_items: Maximum items to store
            ttl_seconds: Time-to-live for items
            redis_client: Optional Redis client for persistence
        """
        self.session_id = session_id or str(uuid4())
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self.redis = redis_client
        self._items: OrderedDict[UUID, ShortTermItem] = OrderedDict()

    def add(
        self,
        content: str,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ShortTermItem:
        """
        Add an item to short-term memory.

        Args:
            content: The content to store
            ttl_seconds: Optional custom TTL
            metadata: Additional metadata

        Returns:
            Created ShortTermItem
        """
        # Clean expired items
        self._cleanup_expired()

        # Evict oldest if at capacity
        while len(self._items) >= self.max_items:
            oldest_key = next(iter(self._items))
            del self._items[oldest_key]

        ttl = ttl_seconds or self.ttl_seconds
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)

        item = ShortTermItem(
            id=uuid4(),
            content=content,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        self._items[item.id] = item

        # Store in Redis if available
        if self.redis:
            key = f"stm:{self.session_id}:{item.id}"
            self.redis.setex(
                key,
                ttl,
                {
                    "content": content,
                    "metadata": metadata or {},
                    "created_at": item.created_at.isoformat(),
                },
            )

        return item

    def get(self, item_id: UUID) -> ShortTermItem | None:
        """Get an item by ID."""
        item = self._items.get(item_id)
        if item and not item.is_expired():
            # Move to end (most recently accessed)
            self._items.move_to_end(item_id)
            return item
        return None

    def get_recent(self, limit: int = 10) -> list[ShortTermItem]:
        """Get most recent items."""
        self._cleanup_expired()
        items = list(self._items.values())
        return items[-limit:][::-1]  # Most recent first

    def get_all(self) -> list[ShortTermItem]:
        """Get all non-expired items."""
        self._cleanup_expired()
        return list(self._items.values())

    def search(self, query: str) -> list[ShortTermItem]:
        """Simple text search in short-term memory."""
        self._cleanup_expired()
        query_lower = query.lower()
        return [
            item
            for item in self._items.values()
            if query_lower in item.content.lower()
        ]

    def remove(self, item_id: UUID) -> bool:
        """Remove an item."""
        if item_id in self._items:
            del self._items[item_id]

            if self.redis:
                self.redis.delete(f"stm:{self.session_id}:{item_id}")

            return True
        return False

    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()

        if self.redis:
            # Delete all keys for this session
            pattern = f"stm:{self.session_id}:*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)

    def _cleanup_expired(self) -> None:
        """Remove expired items."""
        expired = [
            item_id
            for item_id, item in self._items.items()
            if item.is_expired()
        ]
        for item_id in expired:
            del self._items[item_id]

    def to_context(self, limit: int = 10) -> str:
        """
        Export recent memory as context string.

        Useful for including in prompts.
        """
        items = self.get_recent(limit)
        if not items:
            return ""

        lines = ["Recent context:"]
        for item in items:
            lines.append(f"- {item.content}")

        return "\n".join(lines)

    @property
    def count(self) -> int:
        """Get count of non-expired items."""
        self._cleanup_expired()
        return len(self._items)

    @property
    def token_estimate(self) -> int:
        """Estimate token count for all items."""
        self._cleanup_expired()
        total_chars = sum(len(item.content) for item in self._items.values())
        return total_chars // 4  # Rough estimate: 4 chars per token
