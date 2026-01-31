"""
Memory CRUD Routes

Endpoints for creating, reading, updating, and deleting memories.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Query
from pydantic import BaseModel, Field

from agentmem.core.memory import (
    MemoryManager,
    MemoryItem,
    MemoryType,
    MemoryCreate,
    MemoryUpdate,
)
from agentmem.core.compressor import TokenCompressor


router = APIRouter()


class MemoryResponse(BaseModel):
    """Response model for memory data."""

    id: str
    space_id: str
    content: str
    memory_type: str
    importance: float
    metadata: dict
    created_at: str
    accessed_at: str
    access_count: int


class MemoryListResponse(BaseModel):
    """Response model for memory list."""

    memories: list[MemoryResponse]
    total: int
    offset: int
    limit: int


class CreateMemoryRequest(BaseModel):
    """Request body for creating a memory."""

    agent_id: str
    content: str = Field(..., min_length=1)
    memory_type: str = "episodic"
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class CompressRequest(BaseModel):
    """Request body for compression."""

    agent_id: str
    target_ratio: float = Field(default=0.5, ge=0.1, le=1.0)


class CompressResponse(BaseModel):
    """Response from compression."""

    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    memories_processed: int


def get_memory_manager(request: Request) -> MemoryManager:
    """Get memory manager dependency."""
    return MemoryManager(
        db_client=getattr(request.app.state, "db", None),
        embedding_provider=getattr(request.app.state, "embedding_provider", None),
    )


def get_llm_client(request: Request):
    """Get LLM client (Kimi) from app state."""
    return getattr(request.app.state, "llm_client", None)


def memory_to_response(memory: MemoryItem) -> MemoryResponse:
    """Convert MemoryItem to response model."""
    return MemoryResponse(
        id=str(memory.id),
        space_id=str(memory.space_id),
        content=memory.content,
        memory_type=memory.memory_type.value,
        importance=memory.importance,
        metadata=memory.metadata,
        created_at=memory.created_at.isoformat(),
        accessed_at=memory.accessed_at.isoformat(),
        access_count=memory.access_count,
    )


@router.post("", response_model=MemoryResponse)
async def create_memory(
    memory_data: CreateMemoryRequest,
    manager: MemoryManager = Depends(get_memory_manager),
):
    """
    Create a new memory.

    The memory will be stored with embeddings for semantic search.
    """
    # Get or create space for agent
    space = await manager.get_space(UUID(memory_data.agent_id))

    # Determine memory type
    try:
        mem_type = MemoryType(memory_data.memory_type)
    except ValueError:
        mem_type = MemoryType.EPISODIC

    # Store memory
    memory = await manager.store(
        space_id=space.id,
        content=memory_data.content,
        memory_type=mem_type,
        importance=memory_data.importance,
        metadata=memory_data.metadata,
    )

    return memory_to_response(memory)


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    manager: MemoryManager = Depends(get_memory_manager),
):
    """Get a memory by ID."""
    memory = await manager.get(UUID(memory_id))
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    return memory_to_response(memory)


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    updates: MemoryUpdate,
    manager: MemoryManager = Depends(get_memory_manager),
):
    """Update a memory."""
    update_data = updates.model_dump(exclude_unset=True)
    memory = await manager.update(UUID(memory_id), **update_data)

    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    return memory_to_response(memory)


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    manager: MemoryManager = Depends(get_memory_manager),
):
    """Delete a memory."""
    deleted = await manager.delete(UUID(memory_id))

    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"deleted": True, "memory_id": memory_id}


@router.get("/recent", response_model=MemoryListResponse)
async def get_recent_memories(
    agent_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    memory_type: str | None = None,
    manager: MemoryManager = Depends(get_memory_manager),
):
    """Get recent memories for an agent."""
    space = await manager.get_space(UUID(agent_id))

    mem_type = None
    if memory_type:
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            pass

    memories = await manager.list_recent(
        space_id=space.id,
        limit=limit,
        memory_type=mem_type,
    )

    return MemoryListResponse(
        memories=[memory_to_response(m) for m in memories],
        total=len(memories),
        offset=0,
        limit=limit,
    )


@router.post("/compress", response_model=CompressResponse)
async def compress_memories(
    compress_data: CompressRequest,
    request: Request,
    manager: MemoryManager = Depends(get_memory_manager),
):
    """
    Compress memories to reduce token usage.

    This will intelligently summarize and consolidate memories
    while preserving important information.

    Uses Kimi LLM for cost-effective compression.
    """
    space = await manager.get_space(UUID(compress_data.agent_id))

    # Get all memories
    memories = await manager.list_recent(space_id=space.id, limit=1000)

    if not memories:
        return CompressResponse(
            original_tokens=0,
            compressed_tokens=0,
            tokens_saved=0,
            memories_processed=0,
        )

    # Compress with Kimi LLM if available
    llm_client = get_llm_client(request)
    compressor = TokenCompressor(llm_client=llm_client)
    results = await compressor.compress_batch(
        memories,
        target_total_tokens=int(
            sum(compressor.estimate_tokens(m.content) for m in memories)
            * compress_data.target_ratio
        ),
    )

    original_tokens = sum(r.original_tokens for r in results)
    compressed_tokens = sum(r.compressed_tokens for r in results)

    # Update memories with compressed content
    for memory, result in zip(memories, results):
        if result.compressed_content != memory.content:
            await manager.update(
                memory.id,
                content=result.compressed_content,
            )

    return CompressResponse(
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        tokens_saved=original_tokens - compressed_tokens,
        memories_processed=len(memories),
    )


@router.get("/stats/{agent_id}")
async def get_memory_stats(
    agent_id: str,
    manager: MemoryManager = Depends(get_memory_manager),
):
    """Get memory statistics for an agent."""
    space = await manager.get_space(UUID(agent_id))
    count = await manager.count(space.id)

    # Estimate tokens
    memories = await manager.list_recent(space_id=space.id, limit=100)
    compressor = TokenCompressor()
    estimated_tokens = sum(
        compressor.estimate_tokens(m.content) for m in memories
    )

    return {
        "agent_id": agent_id,
        "space_id": str(space.id),
        "memory_count": count,
        "estimated_tokens": estimated_tokens,
        "types": {
            "episodic": len([m for m in memories if m.memory_type == MemoryType.EPISODIC]),
            "semantic": len([m for m in memories if m.memory_type == MemoryType.SEMANTIC]),
            "procedural": len([m for m in memories if m.memory_type == MemoryType.PROCEDURAL]),
        },
    }
