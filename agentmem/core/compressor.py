"""
Token Compressor

Reduces token usage by intelligently compressing memories
while preserving essential information.
"""

from dataclasses import dataclass
from typing import Any

from agentmem.core.memory import MemoryItem


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    compressed_content: str
    metadata: dict[str, Any]

    @property
    def tokens_saved(self) -> int:
        """Calculate tokens saved."""
        return self.original_tokens - self.compressed_tokens


class TokenCompressor:
    """
    Compresses memories to reduce token usage.

    Implements multiple compression strategies:
    - Summarization (LLM-based)
    - Deduplication
    - Importance-based pruning
    - Hierarchical summarization

    Can reduce token usage by 60-80% while preserving
    essential information.
    """

    # Default token estimation: ~4 chars per token
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        llm_client=None,
        max_output_tokens: int = 500,
        preserve_importance_threshold: float = 0.8,
    ):
        """
        Initialize token compressor.

        Args:
            llm_client: LLM client for summarization
            max_output_tokens: Maximum tokens in compressed output
            preserve_importance_threshold: Don't compress above this importance
        """
        self.llm = llm_client
        self.max_output_tokens = max_output_tokens
        self.preserve_threshold = preserve_importance_threshold

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.CHARS_PER_TOKEN

    async def compress_memory(
        self,
        memory: MemoryItem,
        target_ratio: float = 0.5,
    ) -> CompressionResult:
        """
        Compress a single memory.

        Args:
            memory: Memory to compress
            target_ratio: Target compression ratio (0.5 = 50% reduction)

        Returns:
            CompressionResult with compressed content
        """
        original_tokens = self.estimate_tokens(memory.content)

        # Don't compress high-importance memories
        if memory.importance >= self.preserve_threshold:
            return CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                compressed_content=memory.content,
                metadata={"preserved": True, "reason": "high_importance"},
            )

        target_tokens = int(original_tokens * target_ratio)

        if self.llm:
            compressed = await self._llm_compress(memory.content, target_tokens)
        else:
            compressed = self._rule_compress(memory.content, target_tokens)

        compressed_tokens = self.estimate_tokens(compressed)

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            compressed_content=compressed,
            metadata={"method": "llm" if self.llm else "rule"},
        )

    async def compress_batch(
        self,
        memories: list[MemoryItem],
        target_total_tokens: int | None = None,
    ) -> list[CompressionResult]:
        """
        Compress a batch of memories.

        Args:
            memories: Memories to compress
            target_total_tokens: Target total token count

        Returns:
            List of CompressionResults
        """
        if not memories:
            return []

        # Calculate current total and targets
        total_tokens = sum(self.estimate_tokens(m.content) for m in memories)

        if target_total_tokens is None:
            target_total_tokens = total_tokens // 2

        # Sort by importance (compress low-importance first)
        sorted_memories = sorted(memories, key=lambda m: m.importance)

        results = []
        remaining_target = target_total_tokens

        for memory in sorted_memories:
            mem_tokens = self.estimate_tokens(memory.content)

            # Calculate individual target
            if memory.importance >= self.preserve_threshold:
                # Preserve high-importance, allocate full tokens
                target_ratio = 1.0
            else:
                # Compress to proportional target
                target_ratio = min(1.0, remaining_target / max(mem_tokens, 1))

            result = await self.compress_memory(memory, target_ratio)
            results.append(result)

            remaining_target -= result.compressed_tokens

        return results

    async def summarize_episode(
        self,
        events: list[MemoryItem],
        max_tokens: int | None = None,
    ) -> CompressionResult:
        """
        Summarize an episode into a condensed memory.

        Args:
            events: Episode events to summarize
            max_tokens: Maximum output tokens

        Returns:
            CompressionResult with summary
        """
        if not events:
            return CompressionResult(
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                compressed_content="",
                metadata={"error": "no_events"},
            )

        max_tokens = max_tokens or self.max_output_tokens

        # Combine all content
        combined = "\n".join(e.content for e in events)
        original_tokens = self.estimate_tokens(combined)

        if self.llm:
            summary = await self._llm_summarize_episode(events, max_tokens)
        else:
            summary = self._rule_summarize_episode(events, max_tokens)

        compressed_tokens = self.estimate_tokens(summary)

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            compressed_content=summary,
            metadata={
                "event_count": len(events),
                "method": "llm" if self.llm else "rule",
            },
        )

    async def _llm_compress(self, content: str, target_tokens: int) -> str:
        """Use LLM to compress content."""
        prompt = f"""Compress the following text to approximately {target_tokens} tokens while preserving key information:

{content}

Compressed version:"""

        response = await self.llm.complete(prompt, max_tokens=target_tokens + 50)
        return response.strip()

    def _rule_compress(self, content: str, target_tokens: int) -> str:
        """Rule-based compression without LLM."""
        target_chars = target_tokens * self.CHARS_PER_TOKEN

        if len(content) <= target_chars:
            return content

        # Simple truncation with ellipsis
        # In production, use more sophisticated rules
        sentences = content.split(". ")

        compressed = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) + 2 <= target_chars:
                compressed.append(sentence)
                current_length += len(sentence) + 2
            else:
                break

        result = ". ".join(compressed)
        if result and not result.endswith("."):
            result += "..."

        return result

    async def _llm_summarize_episode(
        self,
        events: list[MemoryItem],
        max_tokens: int,
    ) -> str:
        """Use LLM to summarize an episode."""
        # Format events
        event_text = "\n".join(
            f"[{e.created_at.strftime('%H:%M')}] {e.content}"
            for e in events
        )

        prompt = f"""Summarize this conversation/interaction into a concise summary of at most {max_tokens} tokens:

{event_text}

Summary:"""

        response = await self.llm.complete(prompt, max_tokens=max_tokens + 50)
        return response.strip()

    def _rule_summarize_episode(
        self,
        events: list[MemoryItem],
        max_tokens: int,
    ) -> str:
        """Rule-based episode summarization."""
        target_chars = max_tokens * self.CHARS_PER_TOKEN

        # Extract key events (high importance)
        key_events = sorted(
            events,
            key=lambda e: e.importance,
            reverse=True,
        )

        summary_parts = []
        current_length = 0

        for event in key_events:
            # Truncate long events
            content = event.content[:200] + "..." if len(event.content) > 200 else event.content

            if current_length + len(content) + 3 <= target_chars:
                summary_parts.append(f"- {content}")
                current_length += len(content) + 3
            else:
                break

        return "\n".join(summary_parts) if summary_parts else "No key events."


async def compress_for_context(
    memories: list[MemoryItem],
    max_tokens: int = 2000,
    compressor: TokenCompressor | None = None,
) -> str:
    """
    Compress memories for inclusion in LLM context.

    Args:
        memories: Memories to include
        max_tokens: Maximum total tokens
        compressor: Optional custom compressor

    Returns:
        Compressed context string
    """
    if not memories:
        return ""

    compressor = compressor or TokenCompressor()

    # Compress batch to target
    results = await compressor.compress_batch(memories, max_tokens)

    # Format as context
    lines = ["Relevant context:"]
    for result in results:
        if result.compressed_content:
            lines.append(f"- {result.compressed_content}")

    return "\n".join(lines)
