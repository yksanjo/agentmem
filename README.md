# AgentMem

**Agent Memory System** - Persistent memory and state management for AI agents.

Part of the Agent Infrastructure Stack:
- [AgentGate](https://github.com/yksanjo/agentgate) - Authentication
- **AgentMem** (this repo) - Memory/State
- [AgentLens](https://github.com/yksanjo/agentlens) - Observability

## Features

- **Hierarchical Memory** - Working, short-term, and long-term memory tiers
- **Semantic Search** - Vector embeddings for intelligent retrieval
- **Token Compression** - Reduce token usage by 60-80%
- **Episodic Memory** - Conversation and event history
- **Importance Scoring** - Priority-based retention
- **MCP Native** - Model Context Protocol compatible

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yksanjo/agentmem.git
cd agentmem

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SUPABASE_URL="your-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your-key"
export OPENAI_API_KEY="your-key"  # For embeddings

# Run the server
uvicorn api.main:app --reload --port 8001
```

### Using the SDK

```python
from agentmem import Memory

# Initialize with agent ID
memory = Memory(
    agent_id="your-agent-id",
    auth_token="your-token",
)

# Store a memory
await memory.store(
    "User prefers dark mode",
    importance=0.8,
)

# Search memories
results = await memory.search(
    "user preferences",
    limit=5,
)

# Get recent memories
recent = await memory.recent(limit=10)
```

## API Endpoints

### Memory

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/memory` | POST | Create memory |
| `/api/v1/memory/{id}` | GET | Get memory |
| `/api/v1/memory/{id}` | PATCH | Update memory |
| `/api/v1/memory/{id}` | DELETE | Delete memory |
| `/api/v1/memory/recent` | GET | Get recent memories |
| `/api/v1/memory/compress` | POST | Compress memories |
| `/api/v1/memory/stats/{agent_id}` | GET | Get stats |

### Search

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/search` | GET/POST | Semantic search |
| `/api/v1/search/similar/{id}` | GET | Find similar |

### Spaces

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/spaces` | POST | Create space |
| `/api/v1/spaces/{agent_id}` | GET | Get space |
| `/api/v1/spaces/{agent_id}/stats` | GET | Space stats |

### Sync

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sync/sync` | POST | Sync memories |
| `/api/v1/sync/share` | POST | Share memories |

## Memory Types

AgentMem supports three types of memories:

- **Episodic** - Events, conversations, interactions
- **Semantic** - Facts, knowledge, learned information
- **Procedural** - How-to knowledge, processes

## Architecture

```
┌─────────────────────────────────────────┐
│              Agent Code                  │
│  memory = Memory(agent_id="...")        │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│            AgentMem API                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Memory  │  │ Search  │  │  Sync   │ │
│  │  CRUD   │  │ Engine  │  │ Service │ │
│  └─────────┘  └─────────┘  └─────────┘ │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│  pgvector │ │   Redis   │ │  OpenAI   │
│  Storage  │ │   Cache   │ │ Embeddings│
└───────────┘ └───────────┘ └───────────┘
```

## MCP Integration

AgentMem provides an MCP-compatible memory server:

```python
from agentmem.mcp_server.server import create_mcp_server

server = create_mcp_server(memory_manager, semantic_search)

# Available tools:
# - memory_store: Store a new memory
# - memory_search: Semantic search
# - memory_recent: Get recent memories
# - memory_forget: Delete a memory
```

## Database Schema

```sql
-- Memory spaces
CREATE TABLE memory_spaces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Memories with embeddings
CREATE TABLE memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    space_id UUID REFERENCES memory_spaces(id),
    content TEXT NOT NULL,
    embedding vector(1536),
    memory_type VARCHAR(20) DEFAULT 'episodic',
    importance FLOAT DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INT DEFAULT 0
);

-- Vector index
CREATE INDEX ON memories
USING ivfflat (embedding vector_cosine_ops);

-- Similarity search function
CREATE FUNCTION match_memories(
    query_embedding vector(1536),
    match_count int,
    match_threshold float,
    p_space_id uuid
) RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.id,
        m.content,
        m.metadata,
        1 - (m.embedding <=> query_embedding) as similarity
    FROM memories m
    WHERE m.space_id = p_space_id
        AND 1 - (m.embedding <=> query_embedding) > match_threshold
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;
```

## Token Compression

AgentMem includes intelligent compression to reduce context window usage:

```python
from agentmem.core.compressor import TokenCompressor

compressor = TokenCompressor()

# Compress memories for context
result = await compressor.compress_batch(
    memories,
    target_total_tokens=2000,
)

# Summarize an episode
summary = await compressor.summarize_episode(
    events,
    max_tokens=500,
)
```

## Deployment

### Railway

```bash
railway login
railway init
railway up
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Supabase service key |
| `OPENAI_API_KEY` | No | For embeddings |
| `REDIS_URL` | No | Redis for caching |
| `PORT` | No | Server port (default: 8001) |

## License

MIT License - see LICENSE file.
