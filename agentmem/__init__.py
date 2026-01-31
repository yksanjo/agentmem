"""
AgentMem - Agent Memory System

Persistent memory and state management for AI agents with:
- Hierarchical memory (working → short-term → long-term)
- Semantic search and retrieval
- Token-efficient compression
- Cross-session persistence
- MCP-native memory server
"""

__version__ = "0.1.0"


class Memory:
    """
    Main SDK class for AgentMem.

    Provides a simple interface to agent memory with automatic
    tiered storage and retrieval.

    Usage:
        memory = Memory(agent_id="...", auth_token="...")

        # Store a memory
        await memory.store("User prefers dark mode", importance=0.8)

        # Search memories
        results = await memory.search("user preferences", limit=5)

        # Get recent memories
        recent = await memory.recent(limit=10)
    """

    def __init__(
        self,
        agent_id: str,
        auth_token: str | None = None,
        base_url: str = "https://agentmem.railway.app",
    ):
        self.agent_id = agent_id
        self.auth_token = auth_token
        self.base_url = base_url.rstrip("/")

    async def store(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> dict:
        """
        Store a new memory.

        Args:
            content: The memory content
            memory_type: Type of memory (episodic, semantic, procedural)
            importance: Importance score 0-1
            metadata: Additional metadata

        Returns:
            Created memory object
        """
        import httpx

        response = await httpx.AsyncClient().post(
            f"{self.base_url}/api/v1/memory",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            json={
                "agent_id": self.agent_id,
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "metadata": metadata or {},
            },
        )
        response.raise_for_status()
        return response.json()

    async def search(
        self,
        query: str,
        limit: int = 5,
        memory_type: str | None = None,
        min_importance: float = 0.0,
    ) -> list[dict]:
        """
        Semantic search for memories.

        Args:
            query: Search query
            limit: Maximum results
            memory_type: Filter by type
            min_importance: Minimum importance threshold

        Returns:
            List of matching memories
        """
        import httpx

        params = {
            "query": query,
            "limit": limit,
            "agent_id": self.agent_id,
        }
        if memory_type:
            params["memory_type"] = memory_type
        if min_importance > 0:
            params["min_importance"] = min_importance

        response = await httpx.AsyncClient().get(
            f"{self.base_url}/api/v1/search",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            params=params,
        )
        response.raise_for_status()
        return response.json()["memories"]

    async def recent(
        self,
        limit: int = 10,
        memory_type: str | None = None,
    ) -> list[dict]:
        """
        Get recent memories.

        Args:
            limit: Maximum results
            memory_type: Filter by type

        Returns:
            List of recent memories
        """
        import httpx

        params = {"limit": limit, "agent_id": self.agent_id}
        if memory_type:
            params["memory_type"] = memory_type

        response = await httpx.AsyncClient().get(
            f"{self.base_url}/api/v1/memory/recent",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            params=params,
        )
        response.raise_for_status()
        return response.json()["memories"]

    async def forget(
        self,
        memory_id: str,
    ) -> bool:
        """
        Delete a specific memory.

        Args:
            memory_id: The memory ID to delete

        Returns:
            True if deleted
        """
        import httpx

        response = await httpx.AsyncClient().delete(
            f"{self.base_url}/api/v1/memory/{memory_id}",
            headers={"Authorization": f"Bearer {self.auth_token}"},
        )
        return response.status_code == 200

    async def compress(self) -> dict:
        """
        Trigger memory compression to reduce token usage.

        Returns:
            Compression stats
        """
        import httpx

        response = await httpx.AsyncClient().post(
            f"{self.base_url}/api/v1/memory/compress",
            headers={"Authorization": f"Bearer {self.auth_token}"},
            json={"agent_id": self.agent_id},
        )
        response.raise_for_status()
        return response.json()
