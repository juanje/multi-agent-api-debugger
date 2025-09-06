"""
Mock data for the multi-agent system.

This module contains all the mock data used by the agents, centralizing
the mock responses to keep the agent code focused on the LangGraph architecture.
"""

from typing import Dict, List, Any

# API Mock Data
API_JOBS = [
    {
        "id": "job_001",
        "name": "data_processing",
        "status": "completed",
        "template": "basic",
    },
    {
        "id": "job_002",
        "name": "image_analysis",
        "status": "running",
        "template": "premium",
    },
    {
        "id": "job_003",
        "name": "report_generation",
        "status": "failed",
        "template": "standard",
    },
    {
        "id": "job_004",
        "name": "data_validation",
        "status": "pending",
        "template": "basic",
    },
]

API_JOB_RESULTS = {
    "job_001": {
        "job_id": "job_001",
        "status": "completed",
        "output": "Data processing completed successfully",
        "metrics": {"processed_records": 1000, "execution_time": "2.5s"},
    },
    "job_002": {
        "job_id": "job_002",
        "status": "running",
        "output": "Image analysis in progress...",
        "progress": 65,
    },
    "job_003": {
        "job_id": "job_003",
        "status": "failed",
        "error": "Template 'standard' not found in system",
        "error_code": "TEMPLATE_NOT_FOUND",
    },
    "job_004": {
        "job_id": "job_004",
        "status": "pending",
        "output": "Job queued for execution",
    },
}

API_SYSTEM_STATUS = {
    "status": "healthy",
    "active_jobs": 1,
    "queued_jobs": 1,
    "failed_jobs": 1,
    "completed_jobs": 1,
    "system_load": 0.65,
    "last_updated": "2024-01-15T10:30:00Z",
}

# Error Mock Data
ERROR_PATTERNS = {
    "TEMPLATE_NOT_FOUND": {
        "error_code": "TEMPLATE_NOT_FOUND",
        "error_message": "Template 'standard' not found in system",
        "timestamp": "2024-01-15T10:35:00Z",
        "suggestions": ["Use 'basic' template", "Check template availability"],
        "root_cause_hypothesis": "The template file is missing from the expected location, likely due to deployment issues or incorrect path configuration.",
        "confidence_level": "High",
        "severity": "High",
        "recommended_actions": [
            "Verify template name spelling",
            "Check if template exists in template directory",
            "Re-deploy templates from source control",
            "Check file permissions on template directory",
        ],
        "related_components": ["Template Engine", "Deployment System", "File System"],
    },
    "MEMORY_LIMIT_EXCEEDED": {
        "error_code": "MEMORY_LIMIT_EXCEEDED",
        "error_message": "Memory limit exceeded during job execution",
        "timestamp": "2024-01-15T10:35:00Z",
        "suggestions": ["Increase memory allocation", "Optimize job parameters"],
        "root_cause_hypothesis": "The job is consuming more memory than allocated, possibly due to large dataset processing or memory leaks.",
        "confidence_level": "Medium",
        "severity": "High",
        "recommended_actions": [
            "Increase memory limit for job",
            "Optimize data processing algorithm",
            "Check for memory leaks in job code",
            "Consider data chunking for large datasets",
        ],
        "related_components": ["Memory Manager", "Job Executor", "Data Processor"],
    },
    "TIMEOUT_ERROR": {
        "error_code": "TIMEOUT_ERROR",
        "error_message": "Job execution timeout exceeded",
        "timestamp": "2024-01-15T10:35:00Z",
        "suggestions": ["Increase timeout limit", "Optimize job performance"],
        "root_cause_hypothesis": "The job is taking longer than expected to complete, possibly due to performance issues or resource constraints.",
        "confidence_level": "Medium",
        "severity": "Medium",
        "recommended_actions": [
            "Increase job timeout limit",
            "Profile job performance",
            "Check system resource availability",
            "Optimize job algorithm",
        ],
        "related_components": [
            "Job Scheduler",
            "Resource Manager",
            "Performance Monitor",
        ],
    },
    "UNKNOWN_ERROR": {
        "error_code": "UNKNOWN_ERROR",
        "error_message": "An unexpected error occurred",
        "timestamp": "2024-01-15T10:35:00Z",
        "suggestions": ["Check system logs", "Contact support"],
        "root_cause_hypothesis": "An unexpected error occurred that doesn't match known patterns. Further investigation is needed.",
        "confidence_level": "Low",
        "severity": "Medium",
        "recommended_actions": [
            "Check system logs for more details",
            "Reproduce the error if possible",
            "Contact system administrator",
            "Review recent system changes",
        ],
        "related_components": ["Error Handler", "Logging System", "System Monitor"],
    },
}

# Knowledge Base Mock Data
KNOWLEDGE_BASE = [
    {
        "term": "API",
        "content": "RESTful API for job management and execution. Provides endpoints for creating, monitoring, and retrieving job results.",
        "related_topics": ["endpoints", "authentication", "jobs"],
    },
    {
        "term": "jobs",
        "content": "Background processing tasks that can be executed asynchronously. Jobs support different templates and can be monitored for status and results.",
        "related_topics": ["templates", "status", "execution"],
    },
    {
        "term": "authentication",
        "content": "API uses token-based authentication. Include your API token in the Authorization header for all requests.",
        "related_topics": ["tokens", "security", "headers"],
    },
    {
        "term": "templates",
        "content": "Predefined job configurations that determine execution environment and parameters. Available templates: basic, premium, standard.",
        "related_topics": ["configuration", "environment", "parameters"],
    },
    {
        "term": "debugging",
        "content": "Process of analyzing job failures to identify root causes and provide solutions. Includes error analysis and troubleshooting steps.",
        "related_topics": ["errors", "troubleshooting", "analysis"],
    },
    {
        "term": "status",
        "content": "Current state of a job: pending, running, completed, or failed. Use status endpoints to monitor job progress.",
        "related_topics": ["monitoring", "progress", "states"],
    },
]

# Response Templates
RESPONSE_TEMPLATES = {
    "api_success": {
        "title": "✅ API Operation Completed",
        "format": "{title}\n\n{summary}\n\nDetails:\n{details}",
    },
    "api_error": {
        "title": "❌ API Operation Failed",
        "format": "{title}\n\n{summary}\n\nError Details:\n{error_details}\n\nNext Steps:\n{next_steps}",
    },
    "debugging_complete": {
        "title": "🔍 Debugging Analysis Complete",
        "format": "{title}\n\n{analysis_summary}\n\nRoot Cause:\n{root_cause}\n\nRecommended Actions:\n{recommended_actions}",
    },
    "knowledge_response": {
        "title": "📚 Information Found",
        "format": "{title}\n\n{answer}\n\nSource: Knowledge Base",
    },
    "general": {
        "title": "🤖 System Response",
        "format": "{title}\n\n{content}",
    },
}


# Helper functions for mock data access
def get_job_by_id(job_id: str) -> Dict[str, Any]:
    """Get job data by ID."""
    for job in API_JOBS:
        if job["id"] == job_id:
            return job
    return {}


def get_job_result(job_id: str) -> Dict[str, Any]:
    """Get job result by ID."""
    result = API_JOB_RESULTS.get(job_id, {})
    return result if isinstance(result, dict) else {}


def get_error_pattern(error_code: str) -> Dict[str, Any]:
    """Get error pattern by error code."""
    return ERROR_PATTERNS.get(error_code, ERROR_PATTERNS["UNKNOWN_ERROR"])


def search_knowledge(query: str) -> List[Dict[str, Any]]:
    """Search knowledge base for matching terms."""
    query_lower = query.lower()
    results = []

    # Extract key terms from the query (remove common question words and punctuation)
    import string

    question_words = {
        "what",
        "are",
        "is",
        "how",
        "do",
        "does",
        "can",
        "could",
        "would",
        "should",
        "the",
        "a",
        "an",
    }
    # Remove punctuation and split into words
    clean_query = query_lower.translate(str.maketrans("", "", string.punctuation))
    query_terms = [word for word in clean_query.split() if word not in question_words]

    for item in KNOWLEDGE_BASE:
        related_topics = item.get("related_topics", [])

        # Check if any query term matches the item
        matches = False

        # Check term match
        if any(term in str(item["term"]).lower() for term in query_terms):
            matches = True

        # Check content match
        if any(term in str(item["content"]).lower() for term in query_terms):
            matches = True

        # Check related topics match
        if any(
            term in str(topic).lower()
            for topic in related_topics
            for term in query_terms
        ):
            matches = True

        # Also check for full query match (fallback)
        if (
            query_lower in str(item["term"]).lower()
            or query_lower in str(item["content"]).lower()
        ):
            matches = True

        if matches:
            results.append(item)

    return results


def get_response_template(template_type: str) -> Dict[str, str]:
    """Get response template by type."""
    return RESPONSE_TEMPLATES.get(template_type, RESPONSE_TEMPLATES["general"])
