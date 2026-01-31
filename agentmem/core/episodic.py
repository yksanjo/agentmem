"""
Episodic Memory

Event and interaction storage with temporal awareness.
Optimized for storing and retrieving conversation history and events.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from agentmem.core.memory import MemoryItem, MemoryType


@dataclass
class Episode:
    """
    Represents an episode (sequence of related events).

    Episodes group related memories together for
    coherent retrieval of interaction history.
    """

    id: UUID
    space_id: UUID
    name: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[MemoryItem] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        """Check if episode is still active."""
        return self.ended_at is None

    @property
    def duration(self) -> timedelta | None:
        """Get episode duration."""
        if self.ended_at:
            return self.ended_at - self.started_at
        return datetime.utcnow() - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": str(self.id),
            "space_id": str(self.space_id),
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "metadata": self.metadata,
            "event_count": len(self.events),
        }


class EpisodicMemory:
    """
    Episodic memory storage for events and interactions.

    Organizes memories into episodes (e.g., conversations,
    tasks, sessions) for temporal and contextual retrieval.

    Features:
    - Episode-based organization
    - Temporal queries
    - Event sequencing
    - Conversation history
    """

    def __init__(
        self,
        space_id: UUID,
        db_client=None,
        embedding_provider=None,
    ):
        """
        Initialize episodic memory.

        Args:
            space_id: Memory space identifier
            db_client: Database client
            embedding_provider: Provider for embeddings
        """
        self.space_id = space_id
        self.db = db_client
        self.embedding_provider = embedding_provider
        self._episodes: dict[UUID, Episode] = {}
        self._active_episode: Episode | None = None

    async def start_episode(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """
        Start a new episode.

        Args:
            name: Episode name/description
            metadata: Additional metadata

        Returns:
            Created Episode
        """
        # End any active episode
        if self._active_episode:
            await self.end_episode(self._active_episode.id)

        episode = Episode(
            id=uuid4(),
            space_id=self.space_id,
            name=name,
            metadata=metadata or {},
        )

        if self.db:
            self.db.table("episodes").insert(
                {
                    "id": str(episode.id),
                    "space_id": str(episode.space_id),
                    "name": episode.name,
                    "metadata": episode.metadata,
                }
            ).execute()

        self._episodes[episode.id] = episode
        self._active_episode = episode
        return episode

    async def end_episode(self, episode_id: UUID) -> Episode | None:
        """
        End an episode.

        Args:
            episode_id: Episode to end

        Returns:
            Ended episode if found
        """
        episode = self._episodes.get(episode_id)
        if not episode:
            return None

        episode.ended_at = datetime.utcnow()

        if self.db:
            self.db.table("episodes").update(
                {"ended_at": episode.ended_at.isoformat()}
            ).eq("id", str(episode_id)).execute()

        if self._active_episode and self._active_episode.id == episode_id:
            self._active_episode = None

        return episode

    async def add_event(
        self,
        content: str,
        episode_id: UUID | None = None,
        event_type: str = "message",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """
        Add an event to an episode.

        Args:
            content: Event content
            episode_id: Episode to add to (uses active if not specified)
            event_type: Type of event (message, action, observation)
            importance: Importance score
            metadata: Additional metadata

        Returns:
            Created MemoryItem
        """
        # Use active episode if not specified
        if episode_id is None:
            if self._active_episode is None:
                # Auto-start an episode
                await self.start_episode("Auto-created episode")
            episode_id = self._active_episode.id

        episode = self._episodes.get(episode_id)
        if not episode:
            raise ValueError(f"Episode {episode_id} not found")

        # Generate embedding if available
        embedding = None
        if self.embedding_provider:
            embedding = await self.embedding_provider.embed(content)

        event_metadata = metadata or {}
        event_metadata["event_type"] = event_type
        event_metadata["episode_id"] = str(episode_id)

        event = MemoryItem(
            id=uuid4(),
            space_id=self.space_id,
            content=content,
            embedding=embedding,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            metadata=event_metadata,
        )

        if self.db:
            self.db.table("memories").insert(
                {
                    "id": str(event.id),
                    "space_id": str(event.space_id),
                    "content": event.content,
                    "embedding": event.embedding,
                    "memory_type": event.memory_type.value,
                    "importance": event.importance,
                    "metadata": event.metadata,
                }
            ).execute()

        episode.events.append(event)
        return event

    async def get_episode(self, episode_id: UUID) -> Episode | None:
        """Get an episode with all its events."""
        episode = self._episodes.get(episode_id)
        if episode:
            return episode

        if self.db:
            # Fetch episode
            result = (
                self.db.table("episodes")
                .select("*")
                .eq("id", str(episode_id))
                .execute()
            )

            if result.data:
                data = result.data[0]
                episode = Episode(
                    id=UUID(data["id"]),
                    space_id=UUID(data["space_id"]),
                    name=data["name"],
                    started_at=datetime.fromisoformat(data["started_at"]),
                    ended_at=(
                        datetime.fromisoformat(data["ended_at"])
                        if data.get("ended_at")
                        else None
                    ),
                    metadata=data.get("metadata", {}),
                )

                # Fetch events
                events_result = (
                    self.db.table("memories")
                    .select("*")
                    .eq("space_id", str(self.space_id))
                    .contains("metadata", {"episode_id": str(episode_id)})
                    .order("created_at")
                    .execute()
                )

                episode.events = [
                    MemoryItem.from_dict(row) for row in events_result.data
                ]

                self._episodes[episode_id] = episode
                return episode

        return None

    async def list_episodes(
        self,
        limit: int = 10,
        include_events: bool = False,
    ) -> list[Episode]:
        """List recent episodes."""
        if self.db:
            result = (
                self.db.table("episodes")
                .select("*")
                .eq("space_id", str(self.space_id))
                .order("started_at", desc=True)
                .limit(limit)
                .execute()
            )

            episodes = []
            for data in result.data:
                episode = Episode(
                    id=UUID(data["id"]),
                    space_id=UUID(data["space_id"]),
                    name=data["name"],
                    started_at=datetime.fromisoformat(data["started_at"]),
                    ended_at=(
                        datetime.fromisoformat(data["ended_at"])
                        if data.get("ended_at")
                        else None
                    ),
                    metadata=data.get("metadata", {}),
                )

                if include_events:
                    full_episode = await self.get_episode(episode.id)
                    if full_episode:
                        episode = full_episode

                episodes.append(episode)

            return episodes

        return sorted(
            self._episodes.values(),
            key=lambda e: e.started_at,
            reverse=True,
        )[:limit]

    async def get_events_in_range(
        self,
        start_time: datetime,
        end_time: datetime | None = None,
    ) -> list[MemoryItem]:
        """Get events within a time range."""
        end_time = end_time or datetime.utcnow()

        if self.db:
            result = (
                self.db.table("memories")
                .select("*")
                .eq("space_id", str(self.space_id))
                .eq("memory_type", MemoryType.EPISODIC.value)
                .gte("created_at", start_time.isoformat())
                .lte("created_at", end_time.isoformat())
                .order("created_at")
                .execute()
            )

            return [MemoryItem.from_dict(row) for row in result.data]

        events = []
        for episode in self._episodes.values():
            for event in episode.events:
                if start_time <= event.created_at <= end_time:
                    events.append(event)

        return sorted(events, key=lambda e: e.created_at)

    async def get_conversation_history(
        self,
        episode_id: UUID | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Get conversation history formatted for LLM context.

        Args:
            episode_id: Specific episode (uses active if not specified)
            limit: Maximum messages

        Returns:
            List of message dicts with role and content
        """
        if episode_id is None and self._active_episode:
            episode_id = self._active_episode.id

        if episode_id is None:
            return []

        episode = await self.get_episode(episode_id)
        if not episode:
            return []

        messages = []
        for event in episode.events[-limit:]:
            event_type = event.metadata.get("event_type", "message")
            role = event.metadata.get("role", "user")

            if event_type == "message":
                messages.append(
                    {
                        "role": role,
                        "content": event.content,
                    }
                )

        return messages

    @property
    def active_episode(self) -> Episode | None:
        """Get the currently active episode."""
        return self._active_episode
