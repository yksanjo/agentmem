"""
Memory Core

Main memory abstraction layer providing unified access to
different memory tiers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memories."""

    EPISODIC = "episodic"  # Event/interaction memories
    SEMANTIC = "semantic"  # Factual knowledge
    PROCEDURAL = "procedural"  # How-to knowledge
    WORKING = "working"  # Current context


class MemoryCreate(BaseModel):
    """Schema for creating a memory."""

    content: str = Field(..., min_length=1)
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryUpdate(BaseModel):
    """Schema for updating a memory."""

    content: str | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] | None = None


@dataclass
class MemoryItem:
    """
    Represents a single memory item.

    Memories are stored with embeddings for semantic search,
    importance scores for prioritization, and access tracking
    for forgetting curves.
    """

    id: UUID
    space_id: UUID
    content: str
    embedding: list[float] | None = None
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "space_id": str(self.space_id),
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryItem":
        """Create from dictionary representation."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            space_id=(
                UUID(data["space_id"])
                if isinstance(data["space_id"], str)
                else data["space_id"]
            ),
            content=data["content"],
            embedding=data.get("embedding"),
            memory_type=MemoryType(data.get("memory_type", "episodic")),
            importance=data.get("importance", 0.5),
            metadata=data.get("metadata", {}),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if isinstance(data.get("created_at"), str)
                else data.get("created_at", datetime.utcnow())
            ),
            accessed_at=(
                datetime.fromisoformat(data["accessed_at"])
                if isinstance(data.get("accessed_at"), str)
                else data.get("accessed_at", datetime.utcnow())
            ),
            access_count=data.get("access_count", 0),
        )

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


@dataclass
class MemorySpace:
    """
    A memory space for an agent.

    Each agent has its own memory space containing all their memories.
    Spaces can be shared between agents with proper permissions.
    """

    id: UUID
    agent_id: UUID
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "name": self.name,
            "config": self.config,
            "created_at": self.created_at.isoformat(),
        }


class MemoryManager:
    """
    Manages memories across all tiers.

    Provides unified interface for storing, searching, and
    managing memories with automatic tier management.
    """

    def __init__(self, db_client=None, embedding_provider=None):
        """
        Initialize memory manager.

        Args:
            db_client: Database client (Supabase)
            embedding_provider: Provider for generating embeddings
        """
        self.db = db_client
        self.embedding_provider = embedding_provider
        self._spaces: dict[UUID, MemorySpace] = {}
        self._memories: dict[UUID, list[MemoryItem]] = {}  # space_id -> memories

    async def create_space(
        self,
        agent_id: UUID,
        name: str = "default",
        config: dict[str, Any] | None = None,
    ) -> MemorySpace:
        """
        Create a new memory space for an agent.

        Args:
            agent_id: The agent's UUID
            name: Space name
            config: Configuration options

        Returns:
            Created MemorySpace
        """
        space = MemorySpace(
            id=uuid4(),
            agent_id=agent_id,
            name=name,
            config=config or {},
        )

        if self.db:
            self.db.table("memory_spaces").insert(
                {
                    "id": str(space.id),
                    "agent_id": str(space.agent_id),
                    "name": space.name,
                    "config": space.config,
                }
            ).execute()

        self._spaces[space.id] = space
        self._memories[space.id] = []
        return space

    async def get_space(
        self,
        agent_id: UUID,
        name: str = "default",
    ) -> MemorySpace | None:
        """Get or create a memory space for an agent."""
        # Check cache
        for space in self._spaces.values():
            if space.agent_id == agent_id and space.name == name:
                return space

        # Check database
        if self.db:
            result = (
                self.db.table("memory_spaces")
                .select("*")
                .eq("agent_id", str(agent_id))
                .eq("name", name)
                .execute()
            )

            if result.data:
                data = result.data[0]
                space = MemorySpace(
                    id=UUID(data["id"]),
                    agent_id=UUID(data["agent_id"]),
                    name=data["name"],
                    config=data.get("config", {}),
                    created_at=datetime.fromisoformat(data["created_at"]),
                )
                self._spaces[space.id] = space
                return space

        # Create if doesn't exist
        return await self.create_space(agent_id, name)

    async def store(
        self,
        space_id: UUID,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """
        Store a new memory.

        Args:
            space_id: Memory space ID
            content: Memory content
            memory_type: Type of memory
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
            space_id=space_id,
            content=content,
            embedding=embedding,
            memory_type=memory_type,
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

        if space_id not in self._memories:
            self._memories[space_id] = []
        self._memories[space_id].append(memory)

        return memory

    async def get(self, memory_id: UUID) -> MemoryItem | None:
        """Get a memory by ID."""
        # Check cache
        for memories in self._memories.values():
            for memory in memories:
                if memory.id == memory_id:
                    memory.touch()
                    return memory

        # Check database
        if self.db:
            result = (
                self.db.table("memories")
                .select("*")
                .eq("id", str(memory_id))
                .execute()
            )

            if result.data:
                memory = MemoryItem.from_dict(result.data[0])
                memory.touch()

                # Update access time in DB
                self.db.table("memories").update(
                    {"accessed_at": memory.accessed_at.isoformat()}
                ).eq("id", str(memory_id)).execute()

                return memory

        return None

    async def update(
        self,
        memory_id: UUID,
        **updates: Any,
    ) -> MemoryItem | None:
        """Update a memory."""
        memory = await self.get(memory_id)
        if not memory:
            return None

        for key, value in updates.items():
            if hasattr(memory, key) and value is not None:
                setattr(memory, key, value)

        # Regenerate embedding if content changed
        if "content" in updates and self.embedding_provider:
            memory.embedding = await self.embedding_provider.embed(memory.content)

        if self.db:
            self.db.table("memories").update(
                {
                    "content": memory.content,
                    "embedding": memory.embedding,
                    "importance": memory.importance,
                    "metadata": memory.metadata,
                }
            ).eq("id", str(memory_id)).execute()

        return memory

    async def delete(self, memory_id: UUID) -> bool:
        """Delete a memory."""
        # Remove from cache
        for space_id, memories in self._memories.items():
            self._memories[space_id] = [m for m in memories if m.id != memory_id]

        if self.db:
            result = (
                self.db.table("memories")
                .delete()
                .eq("id", str(memory_id))
                .execute()
            )
            return len(result.data) > 0

        return True

    async def list_recent(
        self,
        space_id: UUID,
        limit: int = 10,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryItem]:
        """List recent memories."""
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

        # In-memory
        memories = self._memories.get(space_id, [])
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]

        return sorted(memories, key=lambda m: m.created_at, reverse=True)[:limit]

    async def count(self, space_id: UUID) -> int:
        """Count memories in a space."""
        if self.db:
            result = (
                self.db.table("memories")
                .select("id", count="exact")
                .eq("space_id", str(space_id))
                .execute()
            )
            return result.count or 0

        return len(self._memories.get(space_id, []))
