"""
Search Routes

Endpoints for semantic and temporal memory search.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from agentmem.core.memory import MemoryManager, MemoryType
from agentmem.retrieval.semantic import SemanticSearch, EmbeddingProvider


router = APIRouter()


class SearchRequest(BaseModel):
    """Request body for semantic search."""

    query: str = Field(..., min_length=1)
    agent_id: str
    limit: int = Field(default=10, ge=1, le=50)
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    memory_type: str | None = None
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Single search result."""

    id: str
    content: str
    memory_type: str
    importance: float
    score: float
    created_at: str


class SearchResponse(BaseModel):
    """Response from search."""

    query: str
    memories: list[SearchResult]
    total: int


def get_memory_manager(request: Request) -> MemoryManager:
    """Get memory manager dependency."""
    return MemoryManager(
        db_client=getattr(request.app.state, "db", None),
        embedding_provider=getattr(request.app.state, "embedding_provider", None),
    )


def get_semantic_search(request: Request) -> SemanticSearch:
    """Get semantic search dependency."""
    return SemanticSearch(
        db_client=getattr(request.app.state, "db", None),
        embedding_provider=getattr(request.app.state, "embedding_provider", None),
    )


@router.get("", response_model=SearchResponse)
async def search_memories(
    query: str,
    agent_id: str,
    limit: int = Query(default=10, ge=1, le=50),
    min_score: float = Query(default=0.5, ge=0.0, le=1.0),
    memory_type: str | None = None,
    min_importance: float = Query(default=0.0, ge=0.0, le=1.0),
    manager: MemoryManager = Depends(get_memory_manager),
    search: SemanticSearch = Depends(get_semantic_search),
):
    """
    Semantic search for memories.

    Uses vector embeddings to find memories semantically similar to the query.
    """
    # Get space for agent
    space = await manager.get_space(UUID(agent_id))

    # Parse memory type
    mem_type = None
    if memory_type:
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            pass

    # Perform search
    results = await search.search(
        query=query,
        space_id=space.id,
        limit=limit,
        min_score=min_score,
        memory_type=mem_type,
        min_importance=min_importance,
    )

    return SearchResponse(
        query=query,
        memories=[
            SearchResult(
                id=str(r.memory.id),
                content=r.memory.content,
                memory_type=r.memory.memory_type.value,
                importance=r.memory.importance,
                score=r.score,
                created_at=r.memory.created_at.isoformat(),
            )
            for r in results
        ],
        total=len(results),
    )


@router.post("", response_model=SearchResponse)
async def search_memories_post(
    search_data: SearchRequest,
    manager: MemoryManager = Depends(get_memory_manager),
    search: SemanticSearch = Depends(get_semantic_search),
):
    """
    Semantic search for memories (POST version).

    Same as GET but allows complex query in request body.
    """
    # Get space for agent
    space = await manager.get_space(UUID(search_data.agent_id))

    # Parse memory type
    mem_type = None
    if search_data.memory_type:
        try:
            mem_type = MemoryType(search_data.memory_type)
        except ValueError:
            pass

    # Perform search
    results = await search.search(
        query=search_data.query,
        space_id=space.id,
        limit=search_data.limit,
        min_score=search_data.min_score,
        memory_type=mem_type,
        min_importance=search_data.min_importance,
    )

    return SearchResponse(
        query=search_data.query,
        memories=[
            SearchResult(
                id=str(r.memory.id),
                content=r.memory.content,
                memory_type=r.memory.memory_type.value,
                importance=r.memory.importance,
                score=r.score,
                created_at=r.memory.created_at.isoformat(),
            )
            for r in results
        ],
        total=len(results),
    )


@router.get("/similar/{memory_id}", response_model=SearchResponse)
async def find_similar_memories(
    memory_id: str,
    limit: int = Query(default=5, ge=1, le=20),
    min_score: float = Query(default=0.7, ge=0.0, le=1.0),
    manager: MemoryManager = Depends(get_memory_manager),
    search: SemanticSearch = Depends(get_semantic_search),
):
    """
    Find memories similar to a specific memory.

    Useful for finding related information.
    """
    # Get the reference memory
    memory = await manager.get(UUID(memory_id))
    if not memory:
        return SearchResponse(
            query=f"similar to {memory_id}",
            memories=[],
            total=0,
        )

    # Find similar
    results = await search.find_similar(
        memory=memory,
        limit=limit,
        min_score=min_score,
    )

    return SearchResponse(
        query=f"similar to: {memory.content[:50]}...",
        memories=[
            SearchResult(
                id=str(r.memory.id),
                content=r.memory.content,
                memory_type=r.memory.memory_type.value,
                importance=r.memory.importance,
                score=r.score,
                created_at=r.memory.created_at.isoformat(),
            )
            for r in results
        ],
        total=len(results),
    )
