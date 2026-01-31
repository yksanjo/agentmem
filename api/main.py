"""
AgentMem API

FastAPI application for agent memory management.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import memory, search, spaces, sync


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("AgentMem API starting...")

    # Initialize database connection
    supabase_url = os.environ.get("SUPABASE_URL")
    if supabase_url:
        from supabase import create_client

        supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        app.state.db = create_client(supabase_url, supabase_key)
        print("Connected to Supabase")
    else:
        app.state.db = None
        print("Running without database (in-memory mode)")

    # Initialize embedding provider
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from agentmem.retrieval.semantic import EmbeddingProvider

        app.state.embedding_provider = EmbeddingProvider(
            provider="openai",
            api_key=openai_key,
        )
        print("Using OpenAI embeddings")
    else:
        app.state.embedding_provider = None
        print("Running without embeddings")

    # Initialize LLM client (Kimi for cost savings)
    kimi_key = os.environ.get("KIMI_API_KEY")
    if kimi_key:
        from agentmem.integrations.kimi import create_kimi_client

        app.state.llm_client = create_kimi_client(
            api_key=kimi_key,
            model=os.environ.get("KIMI_MODEL", "moonshot-v1-8k"),
        )
        print("Using Kimi LLM (moonshot)")
    else:
        # Fallback to OpenAI if no Kimi key
        app.state.llm_client = None
        print("Running without LLM client (compression disabled)")

    yield

    # Cleanup LLM client
    if hasattr(app.state, "llm_client") and app.state.llm_client:
        await app.state.llm_client.close()

    print("AgentMem API shutting down...")


app = FastAPI(
    title="AgentMem",
    description="Agent Memory System - Persistent memory for AI agents",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(memory.router, prefix="/api/v1/memory", tags=["memory"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(spaces.router, prefix="/api/v1/spaces", tags=["spaces"])
app.include_router(sync.router, prefix="/api/v1/sync", tags=["sync"])


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "AgentMem",
        "version": "0.1.0",
        "description": "Agent Memory System",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8001)),
        reload=True,
    )
