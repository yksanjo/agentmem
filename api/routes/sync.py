"""
Memory Sync Routes

Endpoints for cross-agent memory synchronization.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from agentmem.core.memory import MemoryManager


router = APIRouter()


class SyncRequest(BaseModel):
    """Request body for memory sync."""

    source_agent_id: str
    target_agent_id: str
    memory_ids: list[str] | None = None
    filter_type: str | None = None
    min_importance: float = Field(default=0.5, ge=0.0, le=1.0)


class SyncResponse(BaseModel):
    """Response from sync operation."""

    synced_count: int
    source_agent_id: str
    target_agent_id: str


class ShareRequest(BaseModel):
    """Request to share specific memories."""

    memory_ids: list[str]
    target_agent_ids: list[str]


class ShareResponse(BaseModel):
    """Response from share operation."""

    shared_count: int
    target_count: int


def get_memory_manager(request: Request) -> MemoryManager:
    """Get memory manager dependency."""
    return MemoryManager(
        db_client=getattr(request.app.state, "db", None),
        embedding_provider=getattr(request.app.state, "embedding_provider", None),
    )


@router.post("/sync", response_model=SyncResponse)
async def sync_memories(
    sync_data: SyncRequest,
    manager: MemoryManager = Depends(get_memory_manager),
):
    """
    Sync memories from one agent to another.

    Copies memories meeting the criteria to the target agent's space.
    Requires appropriate permissions on both agents.
    """
    # Get source space
    source_space = await manager.get_space(UUID(sync_data.source_agent_id))

    # Get target space (create if needed)
    target_space = await manager.get_space(UUID(sync_data.target_agent_id))

    # Get source memories
    memories = await manager.list_recent(
        space_id=source_space.id,
        limit=1000,
    )

    # Filter by importance
    memories = [m for m in memories if m.importance >= sync_data.min_importance]

    # Filter by type if specified
    if sync_data.filter_type:
        memories = [m for m in memories if m.memory_type.value == sync_data.filter_type]

    # Filter by specific IDs if provided
    if sync_data.memory_ids:
        memory_id_set = set(sync_data.memory_ids)
        memories = [m for m in memories if str(m.id) in memory_id_set]

    # Copy memories to target
    synced = 0
    for memory in memories:
        await manager.store(
            space_id=target_space.id,
            content=memory.content,
            memory_type=memory.memory_type,
            importance=memory.importance,
            metadata={
                **memory.metadata,
                "synced_from": str(source_space.id),
                "original_id": str(memory.id),
            },
        )
        synced += 1

    return SyncResponse(
        synced_count=synced,
        source_agent_id=sync_data.source_agent_id,
        target_agent_id=sync_data.target_agent_id,
    )


@router.post("/share", response_model=ShareResponse)
async def share_memories(
    share_data: ShareRequest,
    manager: MemoryManager = Depends(get_memory_manager),
):
    """
    Share specific memories with multiple agents.

    Creates copies of the specified memories in each target agent's space.
    """
    shared_count = 0

    for memory_id in share_data.memory_ids:
        # Get the memory
        memory = await manager.get(UUID(memory_id))
        if not memory:
            continue

        # Share to each target
        for target_id in share_data.target_agent_ids:
            target_space = await manager.get_space(UUID(target_id))

            await manager.store(
                space_id=target_space.id,
                content=memory.content,
                memory_type=memory.memory_type,
                importance=memory.importance,
                metadata={
                    **memory.metadata,
                    "shared_from": str(memory.space_id),
                    "original_id": str(memory.id),
                },
            )
            shared_count += 1

    return ShareResponse(
        shared_count=shared_count,
        target_count=len(share_data.target_agent_ids),
    )
