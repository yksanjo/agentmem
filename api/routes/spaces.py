"""
Memory Space Routes

Endpoints for managing memory spaces.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from agentmem.core.memory import MemoryManager, MemorySpace


router = APIRouter()


class SpaceResponse(BaseModel):
    """Response model for memory space."""

    id: str
    agent_id: str
    name: str
    config: dict
    created_at: str


class CreateSpaceRequest(BaseModel):
    """Request body for creating a space."""

    agent_id: str
    name: str = Field(default="default", min_length=1, max_length=255)
    config: dict = Field(default_factory=dict)


def get_memory_manager(request: Request) -> MemoryManager:
    """Get memory manager dependency."""
    return MemoryManager(
        db_client=getattr(request.app.state, "db", None),
        embedding_provider=getattr(request.app.state, "embedding_provider", None),
    )


@router.post("", response_model=SpaceResponse)
async def create_space(
    space_data: CreateSpaceRequest,
    manager: MemoryManager = Depends(get_memory_manager),
):
    """Create a new memory space for an agent."""
    space = await manager.create_space(
        agent_id=UUID(space_data.agent_id),
        name=space_data.name,
        config=space_data.config,
    )

    return SpaceResponse(
        id=str(space.id),
        agent_id=str(space.agent_id),
        name=space.name,
        config=space.config,
        created_at=space.created_at.isoformat(),
    )


@router.get("/{agent_id}", response_model=SpaceResponse)
async def get_space(
    agent_id: str,
    name: str = "default",
    manager: MemoryManager = Depends(get_memory_manager),
):
    """Get or create a memory space for an agent."""
    space = await manager.get_space(UUID(agent_id), name)

    return SpaceResponse(
        id=str(space.id),
        agent_id=str(space.agent_id),
        name=space.name,
        config=space.config,
        created_at=space.created_at.isoformat(),
    )


@router.get("/{agent_id}/stats")
async def get_space_stats(
    agent_id: str,
    name: str = "default",
    manager: MemoryManager = Depends(get_memory_manager),
):
    """Get statistics for a memory space."""
    space = await manager.get_space(UUID(agent_id), name)
    count = await manager.count(space.id)

    return {
        "space_id": str(space.id),
        "agent_id": agent_id,
        "name": space.name,
        "memory_count": count,
    }
