"""
Vector store service for Long Term Memory using ChromaDB.

This module provides the low-level interface to the vector database
for storing and retrieving memory entries.
"""

from __future__ import annotations
import asyncio
import logging
import os
from typing import List, Optional, Dict, Any
from uuid import uuid4

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from .schema import MemoryEntry, MemoryType, SearchResult

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Vector store service using ChromaDB for semantic search and storage.

    Handles the storage and retrieval of memory entries using embeddings
    for semantic similarity search.
    """

    def __init__(self, persist_directory: str = "./data/ltm"):
        """
        Initialize the vector store service.

        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._initialized = False

        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)

    async def _initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        if self._initialized or not CHROMADB_AVAILABLE:
            return

        try:
            # Initialize in a thread to avoid blocking
            def init_chroma():
                client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
                collection = client.get_or_create_collection(
                    name="ltm_memories",
                    metadata={"description": "Long Term Memory for multi-agent system"},
                )
                # Initialize embedding model
                model = SentenceTransformer("all-MiniLM-L6-v2")
                return client, collection, model

            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            (
                self._client,
                self._collection,
                self._embedding_model,
            ) = await loop.run_in_executor(None, init_chroma)

            self._initialized = True
            logger.info(
                f"Initialized ChromaDB vector store at {self.persist_directory}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self._initialized = False

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using sentence transformers."""
        if not self._embedding_model:
            raise RuntimeError("Embedding model not initialized")
        return self._embedding_model.encode(text).tolist()

    async def store_memory(self, memory: MemoryEntry) -> str:
        """
        Store a memory entry in the vector database.

        Args:
            memory: The memory entry to store

        Returns:
            The ID of the stored memory
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, skipping memory storage")
            return "mock-id"

        await self._initialize()
        if not self._initialized:
            logger.error("Vector store not initialized")
            return "error-id"

        try:
            # Generate ID if not provided
            memory_id = memory.id or str(uuid4())
            memory.id = memory_id

            # Get searchable text and embedding
            searchable_text = memory.get_searchable_text()

            def store_in_thread():
                embedding = self._get_embedding(searchable_text)

                # Prepare metadata with query and response for better reconstruction
                metadata = memory.get_metadata_dict()
                metadata["user_query"] = memory.user_query
                metadata["system_response"] = memory.system_response

                # Store in ChromaDB
                self._collection.add(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[searchable_text],
                    metadatas=[metadata],
                )
                return memory_id

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, store_in_thread)

            logger.debug(f"Stored memory {memory_id} of type {memory.type}")
            return result

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return "error-id"

    async def search_memories(
        self, query: str, memory_type: Optional[MemoryType] = None, limit: int = 5
    ) -> List[SearchResult]:
        """
        Search for relevant memories by semantic similarity.

        Args:
            query: The search query
            memory_type: Optional filter by memory type
            limit: Maximum number of results to return

        Returns:
            List of search results with similarity scores
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, returning empty results")
            return []

        await self._initialize()
        if not self._initialized:
            logger.error("Vector store not initialized")
            return []

        try:

            def search_in_thread():
                # Get query embedding
                query_embedding = self._get_embedding(query)

                # Build where clause for filtering
                where_clause = {}
                if memory_type:
                    where_clause["type"] = memory_type.value

                # Search in ChromaDB
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=where_clause if where_clause else None,
                    include=["documents", "metadatas", "distances"],
                )

                return results

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, search_in_thread)

            # Convert results to SearchResult objects
            search_results = []
            if results["ids"] and results["ids"][0]:  # Check if we have results
                for i, memory_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    document = results["documents"][0][i]
                    distance = (
                        results["distances"][0][i] if results["distances"] else 0.0
                    )
                    similarity_score = 1.0 - distance  # Convert distance to similarity

                    # Reconstruct memory entry from metadata
                    memory = self._reconstruct_memory_from_metadata(
                        memory_id, document, metadata
                    )

                    search_results.append(
                        SearchResult(
                            memory=memory, similarity_score=similarity_score, rank=i + 1
                        )
                    )

            logger.debug(f"Found {len(search_results)} memories for query: {query}")
            return search_results

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    def _reconstruct_memory_from_metadata(
        self, memory_id: str, document: str, metadata: Dict[str, Any]
    ) -> MemoryEntry:
        """Reconstruct a MemoryEntry from ChromaDB metadata."""
        # Extract basic fields
        memory_type = MemoryType(metadata.get("type", "qa_session"))

        # Use stored query and response from metadata, with fallbacks
        user_query = metadata.get("user_query", document)
        system_response = metadata.get("system_response", "Stored response")

        # Legacy support: if metadata doesn't have user_query/system_response,
        # try to split the document (for backwards compatibility)
        if not metadata.get("user_query") and " | " in document:
            parts = document.split(" | ", 2)
            if len(parts) >= 2:
                user_query, system_response = parts[0], parts[1]

        return MemoryEntry(
            id=memory_id,
            type=memory_type,
            timestamp=metadata.get("timestamp", ""),
            user_query=user_query,
            system_response=system_response,
            context={},
            metadata=metadata,
            error_code=metadata.get("error_code"),
            confidence_level=metadata.get("confidence_level"),
            severity=metadata.get("severity"),
            api_operation=metadata.get("api_operation"),
            success=metadata.get("success"),
            related_topics=metadata.get("related_topics", []),
        )

    async def get_recent_memories(
        self, memory_type: Optional[MemoryType] = None, limit: int = 5
    ) -> List[MemoryEntry]:
        """
        Get recent memories of a specific type.

        Args:
            memory_type: Optional filter by memory type
            limit: Maximum number of results to return

        Returns:
            List of recent memory entries
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, returning empty results")
            return []

        await self._initialize()
        if not self._initialized:
            logger.error("Vector store not initialized")
            return []

        try:

            def get_recent_in_thread():
                where_clause = {}
                if memory_type:
                    where_clause["type"] = memory_type.value

                # Get all results and sort by timestamp
                results = self._collection.get(
                    where=where_clause if where_clause else None,
                    include=["documents", "metadatas"],
                )

                return results

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, get_recent_in_thread)

            # Convert and sort by timestamp
            memories = []
            if results["ids"]:
                for i, memory_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    document = results["documents"][i]

                    memory = self._reconstruct_memory_from_metadata(
                        memory_id, document, metadata
                    )
                    memories.append(memory)

            # Sort by timestamp (most recent first) and limit
            memories.sort(key=lambda m: m.timestamp, reverse=True)
            return memories[:limit]

        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []

    async def get_memory_count(self) -> int:
        """Get total count of stored memories."""
        if not CHROMADB_AVAILABLE:
            return 0

        await self._initialize()
        if not self._initialized:
            return 0

        try:

            def count_in_thread():
                return self._collection.count()

            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, count_in_thread)
            return count

        except Exception as e:
            logger.error(f"Failed to get memory count: {e}")
            return 0
