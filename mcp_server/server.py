"""
MCP Memory Server

Model Context Protocol compatible memory server.
Provides memory capabilities to MCP-compatible LLM tools.
"""

import asyncio
import json
from typing import Any
from uuid import UUID

# MCP Server implementation
# This provides an MCP-compatible interface for memory operations


class MCPMemoryServer:
    """
    MCP-compatible memory server.

    Exposes memory operations as MCP tools:
    - memory_store: Store a new memory
    - memory_search: Semantic search
    - memory_recent: Get recent memories
    - memory_forget: Delete a memory
    """

    def __init__(
        self,
        memory_manager=None,
        semantic_search=None,
    ):
        """
        Initialize MCP memory server.

        Args:
            memory_manager: MemoryManager instance
            semantic_search: SemanticSearch instance
        """
        self.memory_manager = memory_manager
        self.semantic_search = semantic_search

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Get MCP tool definitions.

        Returns:
            List of tool definitions in MCP format
        """
        return [
            {
                "name": "memory_store",
                "description": "Store a new memory for later retrieval",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory content to store",
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score 0-1 (default 0.5)",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural"],
                            "description": "Type of memory (default: episodic)",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "memory_search",
                "description": "Search memories semantically by query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default 5)",
                            "minimum": 1,
                            "maximum": 20,
                        },
                        "min_importance": {
                            "type": "number",
                            "description": "Minimum importance threshold",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_recent",
                "description": "Get recent memories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results (default 10)",
                            "minimum": 1,
                            "maximum": 50,
                        },
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural"],
                            "description": "Filter by memory type",
                        },
                    },
                },
            },
            {
                "name": "memory_forget",
                "description": "Delete a specific memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "ID of the memory to delete",
                        },
                    },
                    "required": ["memory_id"],
                },
            },
        ]

    async def handle_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_id: str,
    ) -> dict[str, Any]:
        """
        Handle an MCP tool call.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            agent_id: ID of the calling agent

        Returns:
            Tool result
        """
        if tool_name == "memory_store":
            return await self._store_memory(agent_id, arguments)
        elif tool_name == "memory_search":
            return await self._search_memory(agent_id, arguments)
        elif tool_name == "memory_recent":
            return await self._recent_memory(agent_id, arguments)
        elif tool_name == "memory_forget":
            return await self._forget_memory(arguments)
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _store_memory(
        self,
        agent_id: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Store a memory."""
        from agentmem.core.memory import MemoryType

        space = await self.memory_manager.get_space(UUID(agent_id))

        memory_type = MemoryType.EPISODIC
        if args.get("memory_type"):
            try:
                memory_type = MemoryType(args["memory_type"])
            except ValueError:
                pass

        memory = await self.memory_manager.store(
            space_id=space.id,
            content=args["content"],
            memory_type=memory_type,
            importance=args.get("importance", 0.5),
        )

        return {
            "stored": True,
            "memory_id": str(memory.id),
            "content": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
        }

    async def _search_memory(
        self,
        agent_id: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Search memories."""
        space = await self.memory_manager.get_space(UUID(agent_id))

        results = await self.semantic_search.search(
            query=args["query"],
            space_id=space.id,
            limit=args.get("limit", 5),
            min_importance=args.get("min_importance", 0.0),
        )

        return {
            "query": args["query"],
            "results": [
                {
                    "id": str(r.memory.id),
                    "content": r.memory.content,
                    "score": r.score,
                    "importance": r.memory.importance,
                }
                for r in results
            ],
            "count": len(results),
        }

    async def _recent_memory(
        self,
        agent_id: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Get recent memories."""
        from agentmem.core.memory import MemoryType

        space = await self.memory_manager.get_space(UUID(agent_id))

        memory_type = None
        if args.get("memory_type"):
            try:
                memory_type = MemoryType(args["memory_type"])
            except ValueError:
                pass

        memories = await self.memory_manager.list_recent(
            space_id=space.id,
            limit=args.get("limit", 10),
            memory_type=memory_type,
        )

        return {
            "memories": [
                {
                    "id": str(m.id),
                    "content": m.content,
                    "type": m.memory_type.value,
                    "importance": m.importance,
                    "created_at": m.created_at.isoformat(),
                }
                for m in memories
            ],
            "count": len(memories),
        }

    async def _forget_memory(
        self,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Delete a memory."""
        deleted = await self.memory_manager.delete(UUID(args["memory_id"]))

        return {
            "deleted": deleted,
            "memory_id": args["memory_id"],
        }


def create_mcp_server(memory_manager, semantic_search):
    """
    Create an MCP memory server.

    This returns a server instance that can be used with
    MCP-compatible tools and frameworks.

    Args:
        memory_manager: MemoryManager instance
        semantic_search: SemanticSearch instance

    Returns:
        MCPMemoryServer instance
    """
    return MCPMemoryServer(
        memory_manager=memory_manager,
        semantic_search=semantic_search,
    )


# Standalone server runner
if __name__ == "__main__":
    import sys
    import os

    # Add parent to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agentmem.core.memory import MemoryManager
    from agentmem.retrieval.semantic import SemanticSearch

    async def main():
        """Run the MCP server."""
        # Initialize components
        manager = MemoryManager()
        search = SemanticSearch()

        # Create server
        server = create_mcp_server(manager, search)

        # Print tool definitions
        print("MCP Memory Server Tools:")
        for tool in server.get_tools():
            print(f"  - {tool['name']}: {tool['description']}")

        print("\nServer ready. Listening for MCP requests...")

        # In production, this would start an actual MCP server
        # For now, just demonstrate the interface

    asyncio.run(main())
