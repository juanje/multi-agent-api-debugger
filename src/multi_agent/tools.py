"""
Tools and utilities for the multi-agent system.
"""

import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


# Detection patterns
CALC_RE = re.compile(r"(\d+)\s*([+\-*/x])\s*(\d+)")

ROUTE_TOOLS_RE = re.compile(
    r"\b(calc|sum|subtract|multiply|divide|\d+\s*[+\-*/x]\s*\d+)\b", re.I
)
ROUTE_RAG_RE = re.compile(
    r"\b(doc|document|manual|paper|article|context|source|reference|rag)\b",
    re.I,
)

# Patterns for multiple operations detection
PARALLEL_PATTERNS = [
    r"(\d+\s*[+\-*/x]\s*\d+)\s+and\s+(\d+\s*[+\-*/x]\s*\d+)",  # "2+3 and 8/2"
    r"(\d+\s*[+\-*/x]\s*\d+)\s+and\s+(time|clock)",  # "2+3 and time"
    r"(time|clock)\s+and\s+(\d+\s*[+\-*/x]\s*\d+)",  # "time and 2+3"
    r"(time|clock)\s+and\s+(doc|document|manual)",  # "time and document"
    r"(doc|document|manual)\s+and\s+(time|clock)",  # "document and time"
    r"(tell\s+me\s+the\s+time)\s+and\s+(my\s+tasks|tasks|search\s+the\s+manual|manual)",  # "tell me the time and my tasks"
    r"(my\s+tasks|tasks|search\s+the\s+manual|manual)\s+and\s+(tell\s+me\s+the\s+time)",  # "my tasks and tell me the time"
]

SEQUENTIAL_PATTERNS = [
    r"sum\s+of\s+(\d+\s*[+\-*/x]\s*\d+)\s+and\s+(\d+\s*[+\-*/x]\s*\d+)",  # "sum of 3x8 and 129/3"
    r"search\s+for\s+information\s+about\s+(\w+)\s+for\s+this\s+time",  # "search for information about tasks for this time"
    r"calculate\s+(\d+\s*[+\-*/x]\s*\d+)\s+and\s+then\s+search\s+for\s+(\w+)",  # "calculate 5*3 and then search for info"
]


def tool_calculator(text: str) -> str:
    """Simple calculator for basic mathematical operations."""
    m = CALC_RE.search(text)
    if not m:
        return "I didn't find any simple operation."
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "+":
        res: Union[int, float] = a + b
    elif op == "-":
        res = a - b
    elif op in ["*", "x"]:
        res = a * b
    else:  # '/'
        if b != 0:
            res = a / b
        else:
            return f"Result (mock calc): {a}{op}{b} = ∞"
    return f"Result (mock calc): {a}{op}{b} = {res}"


def tool_time(_: str) -> str:
    """Tool to query the current time."""
    return f"Current time (mock): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def mock_retrieve(query: str) -> list[dict]:
    """Simulates document retrieval."""
    return [
        {"title": "Doc 1", "snippet": f"Brief summary about '{query}' (part 1)."},
        {"title": "Doc 2", "snippet": f"Related notes about '{query}' (part 2)."},
    ]


def mock_rag_answer(query: str, docs: list[dict]) -> str:
    """Simulates RAG answer based on documents."""
    joined = " ".join(d["snippet"] for d in docs)
    return f"(RAG-mock) Brief answer to: '{query}'. Context: {joined}"


def detect_parallel_operations(text: str) -> Optional[List[str]]:
    """Detects if there are operations that can be executed in parallel."""
    for pattern in PARALLEL_PATTERNS:
        match = re.search(pattern, text, re.I)
        if match:
            operations = []
            for group in match.groups():
                if group and group.strip():
                    # Normalize operations
                    op = group.strip().lower()
                    if "time" in op or "clock" in op:
                        operations.append("time")
                    elif "doc" in op or "manual" in op or "tasks" in op:
                        operations.append("rag")
                    else:
                        operations.append(group.strip())
            return operations
    return None


def detect_sequential_operations(text: str) -> Optional[List[Dict[str, Any]]]:
    """Detects if there are operations that should be executed sequentially."""
    for pattern in SEQUENTIAL_PATTERNS:
        match = re.search(pattern, text, re.I)
        if match:
            steps = []
            groups = match.groups()

            if "sum of" in text:
                # For "sum of 3x8 and 129/3"
                for i, group in enumerate(groups):
                    if group and group.strip():
                        steps.append(
                            {"type": "calc", "operation": group.strip(), "step": i + 1}
                        )
                # Add final sum step
                steps.append({"type": "sum", "step": len(steps) + 1})
            elif "search for information" in text:
                # For "search for information about X for this time"
                steps.append({"type": "time", "step": 1})
                steps.append(
                    {
                        "type": "rag",
                        "query": groups[0] if groups else "tasks",
                        "step": 2,
                    }
                )
            elif "calculate" in text and "then search" in text:
                # For "calculate X and then search for Y"
                steps.append(
                    {
                        "type": "calc",
                        "operation": groups[0] if groups else "",
                        "step": 1,
                    }
                )
                steps.append(
                    {
                        "type": "rag",
                        "query": groups[1] if len(groups) > 1 else "",
                        "step": 2,
                    }
                )

            return steps
    return None
