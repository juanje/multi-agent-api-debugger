"""
Long Term Memory (LTM) service for the multi-agent system.

This module provides the high-level interface for storing and retrieving
memories, abstracting the vector store implementation details.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any

from .schema import MemoryEntry, MemoryType, SearchResult
from .vector_store import VectorStoreService
from ..graph.state import GraphState

logger = logging.getLogger(__name__)

# Global LTM service instance
_ltm_service: Optional[LTMService] = None


class LTMService:
    """
    Long Term Memory service for storing and retrieving system interactions.

    This service provides high-level methods for different types of memory
    operations that agents can use to store and retrieve knowledge.
    """

    def __init__(self, persist_directory: str = "./data/ltm"):
        """
        Initialize the LTM service.

        Args:
            persist_directory: Directory to persist the vector database
        """
        self.vector_store = VectorStoreService(persist_directory)

    async def store_qa_session(
        self,
        user_query: str,
        system_response: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a Q&A session in long term memory.

        Args:
            user_query: The user's original query
            system_response: The system's response
            context: Additional context information
            metadata: Additional metadata

        Returns:
            The ID of the stored memory
        """
        memory = MemoryEntry(
            type=MemoryType.QA_SESSION,
            user_query=user_query,
            system_response=system_response,
            context=context or {},
            metadata=metadata or {},
        )

        memory_id = await self.vector_store.store_memory(memory)
        logger.info(f"Stored Q&A session: {memory_id}")
        return memory_id

    async def store_debug_analysis(
        self,
        analysis: Dict[str, Any],
        user_query: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store debugging analysis in long term memory.

        Args:
            analysis: The root cause analysis data
            user_query: The user's original query
            response: The formatted response
            context: Additional context information

        Returns:
            The ID of the stored memory
        """
        memory = MemoryEntry(
            type=MemoryType.DEBUG_ANALYSIS,
            user_query=user_query,
            system_response=response,
            context=context or {},
            metadata=analysis,
            error_code=analysis.get("error_code"),
            confidence_level=analysis.get("confidence_level"),
            severity=analysis.get("severity"),
        )

        memory_id = await self.vector_store.store_memory(memory)
        logger.info(
            f"Stored debug analysis: {memory_id} for error {analysis.get('error_code', 'unknown')}"
        )
        return memory_id

    async def store_api_operation(
        self,
        operation: str,
        user_query: str,
        response: str,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store API operation result in long term memory.

        Args:
            operation: The API operation performed
            user_query: The user's original query
            response: The formatted response
            success: Whether the operation was successful
            context: Additional context information
            metadata: Additional metadata

        Returns:
            The ID of the stored memory
        """
        memory = MemoryEntry(
            type=MemoryType.API_OPERATION,
            user_query=user_query,
            system_response=response,
            context=context or {},
            metadata=metadata or {},
            api_operation=operation,
            success=success,
        )

        memory_id = await self.vector_store.store_memory(memory)
        logger.info(
            f"Stored API operation: {memory_id} for {operation} (success: {success})"
        )
        return memory_id

    async def store_knowledge_query(
        self,
        user_query: str,
        response: str,
        topics: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store knowledge query and answer in long term memory.

        Args:
            user_query: The user's original query
            response: The knowledge assistant's response
            topics: Related topics
            context: Additional context information
            metadata: Additional metadata

        Returns:
            The ID of the stored memory
        """
        memory = MemoryEntry(
            type=MemoryType.KNOWLEDGE_QUERY,
            user_query=user_query,
            system_response=response,
            context=context or {},
            metadata=metadata or {},
            related_topics=topics or [],
        )

        memory_id = await self.vector_store.store_memory(memory)
        logger.info(f"Stored knowledge query: {memory_id}")
        return memory_id

    async def search_similar_errors(
        self, error_code: str, error_message: str, limit: int = 3
    ) -> List[SearchResult]:
        """
        Search for similar error patterns in long term memory.

        Args:
            error_code: The error code to search for
            error_message: The error message to search for
            limit: Maximum number of results to return

        Returns:
            List of similar debug analyses
        """
        # Create search query combining error code and message
        search_query = f"{error_code} {error_message}"

        results = await self.vector_store.search_memories(
            query=search_query, memory_type=MemoryType.DEBUG_ANALYSIS, limit=limit
        )

        logger.info(f"Found {len(results)} similar errors for {error_code}")
        return results

    async def search_knowledge(self, query: str, limit: int = 5) -> List[SearchResult]:
        """
        Search for relevant Q&A sessions and knowledge in long term memory.

        Args:
            query: The search query
            limit: Maximum number of results to return

        Returns:
            List of relevant knowledge entries
        """
        # Search both Q&A sessions and knowledge queries
        qa_results = await self.vector_store.search_memories(
            query=query, memory_type=MemoryType.QA_SESSION, limit=limit // 2
        )

        knowledge_results = await self.vector_store.search_memories(
            query=query, memory_type=MemoryType.KNOWLEDGE_QUERY, limit=limit // 2
        )

        # Combine and sort by similarity score
        all_results = qa_results + knowledge_results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)

        logger.info(f"Found {len(all_results)} relevant knowledge entries for query")
        return all_results[:limit]

    async def search_api_operations(
        self, operation: str, success_only: bool = True, limit: int = 5
    ) -> List[SearchResult]:
        """
        Search for similar API operations in long term memory.

        Args:
            operation: The API operation to search for
            success_only: Whether to return only successful operations
            limit: Maximum number of results to return

        Returns:
            List of similar API operations
        """
        results = await self.vector_store.search_memories(
            query=operation,
            memory_type=MemoryType.API_OPERATION,
            limit=limit * 2,  # Get more to filter
        )

        # Filter by success if requested
        if success_only:
            results = [r for r in results if r.memory.success is True]

        logger.info(f"Found {len(results)} similar API operations for {operation}")
        return results[:limit]

    async def get_recent_interactions(self, limit: int = 10) -> List[MemoryEntry]:
        """
        Get recent interactions across all memory types.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of recent memory entries
        """
        memories = await self.vector_store.get_recent_memories(limit=limit)
        logger.info(f"Retrieved {len(memories)} recent interactions")
        return memories

    async def store_from_state(self, state: GraphState) -> List[str]:
        """
        Store relevant memories from the current graph state.

        This method analyzes the graph state and stores appropriate
        memories based on the type of interaction that occurred.

        Args:
            state: The current graph state

        Returns:
            List of memory IDs that were stored
        """
        stored_ids: List[str] = []

        # Get user query from messages
        # Strategy: Find the first message that doesn't start with agent emojis
        # This heuristic works well because:
        # 1. User messages are plain text without emoji prefixes
        # 2. Agent messages always start with specific emojis (🎯, 🔧, etc.)
        # 3. The first non-emoji message is typically the original user input
        # Alternative approaches considered:
        # - Track HumanMessage types: Would require LangGraph message tracking
        # - Use message roles: Current architecture doesn't use role-based filtering
        # - Parse message metadata: Not consistently available across all flows
        user_query = ""
        messages = state.get("messages", [])
        if messages:
            for msg in messages:
                content = getattr(msg, "content", "")
                if isinstance(content, str) and not content.startswith(
                    ("🎯", "🔧", "📋", "✅", "❌", "🔍", "📚")
                ):
                    user_query = content
                    break

        if not user_query:
            logger.warning("No user query found in state, skipping LTM storage")
            return stored_ids

        final_response = state.get("final_response", "")
        if not final_response:
            logger.warning("No final response found in state, skipping LTM storage")
            return stored_ids

        # Store debugging analysis if present
        root_cause_analysis = state.get("root_cause_analysis")
        if root_cause_analysis:
            memory_id = await self.store_debug_analysis(
                analysis=root_cause_analysis,
                user_query=user_query,
                response=final_response,
                context={
                    "todo_list": state.get("todo_list", []),
                    "error_info": state.get("error_info", {}),
                },
            )
            stored_ids.append(memory_id)

        # Store API operations if present
        results = state.get("results", {})
        if results:
            for operation, result in results.items():
                success = "error" not in str(result).lower()
                memory_id = await self.store_api_operation(
                    operation=operation,
                    user_query=user_query,
                    response=final_response,
                    success=success,
                    context={"result": result},
                )
                stored_ids.append(memory_id)

        # Store knowledge query if it was a knowledge-based interaction
        if not root_cause_analysis and not results:
            # This was likely a knowledge query
            knowledge_summary = state.get("knowledge_summary", [])
            memory_id = await self.store_knowledge_query(
                user_query=user_query,
                response=final_response,
                context={"route": state.get("route", "")},
                metadata={"knowledge_summary": knowledge_summary},
            )
            stored_ids.append(memory_id)

        # Always store as Q&A session for general reference
        memory_id = await self.store_qa_session(
            user_query=user_query,
            system_response=final_response,
            context={
                "route": state.get("route", ""),
                "has_errors": bool(state.get("error_info")),
                "has_analysis": bool(root_cause_analysis),
                "has_results": bool(results),
            },
        )
        stored_ids.append(memory_id)

        logger.info(f"Stored {len(stored_ids)} memories from graph state")
        return stored_ids

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        total_count = await self.vector_store.get_memory_count()

        return {
            "total_memories": total_count,
            "vector_store_initialized": self.vector_store._initialized,
        }


def get_ltm_service() -> LTMService:
    """Get or create the global LTM service instance."""
    global _ltm_service
    if _ltm_service is None:
        _ltm_service = LTMService()
    return _ltm_service
