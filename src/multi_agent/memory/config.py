"""
Configuration constants for the Long Term Memory system.
"""

# Similarity thresholds for LTM analysis
LTM_CONFIDENCE_THRESHOLD = (
    0.8  # Minimum similarity score to boost confidence in debugger
)
LTM_SEARCH_LIMIT_DEFAULT = 5  # Default limit for search results
LTM_RECENT_LIMIT_DEFAULT = 10  # Default limit for recent memories

# ChromaDB configuration
CHROMA_COLLECTION_NAME = "ltm_memories"
CHROMA_MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer model

# LTM persistence settings
LTM_DEFAULT_DIRECTORY = "./data/ltm"
