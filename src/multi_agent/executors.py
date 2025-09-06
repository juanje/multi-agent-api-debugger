"""
Executors for parallel and sequential operations.
"""

import asyncio
import re
from langchain_core.messages import AIMessage

from .state import ChatState
from .tools import tool_calculator, tool_time, mock_retrieve, mock_rag_answer


async def execute_operation_async(operation: str) -> str:
    """Executes an operation asynchronously."""
    from .tools import CALC_RE

    if CALC_RE.search(operation):
        return tool_calculator(operation)
    elif "time" in operation.lower():
        return tool_time(operation)
    elif operation.lower() == "rag" or any(
        keyword in operation.lower()
        for keyword in ["doc", "document", "manual", "tasks"]
    ):
        # For RAG operations, use a generic query
        query = "manual" if operation.lower() == "rag" else operation
        docs = mock_retrieve(query)
        return mock_rag_answer(query, docs)
    else:
        return f"Unrecognized operation: {operation}"


def parallel_executor_node(state: ChatState) -> ChatState:
    """Executes operations in parallel using asyncio."""
    operations = state.get("operations", [])
    if not operations:
        return state

    # Execute operations in parallel
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(
            asyncio.gather(*[execute_operation_async(op) for op in operations])
        )
    finally:
        loop.close()

    # Create structured response
    new_state = state.copy()
    msgs = list(new_state["messages"])

    # Show intermediate steps
    intermediate_steps = []
    for i, (op, result) in enumerate(zip(operations, results), 1):
        intermediate_steps.append(f"  {i}. {op} → {result}")

    # Show intermediate steps
    intermediate_response = f"""🔄 Executing operations in parallel...

{chr(10).join(intermediate_steps)}"""

    # Final consolidated result - cleaner format
    if len(results) == 2:
        # For two operations, use "and" instead of "|"
        final_result = (
            f"{results[0].split(' = ')[-1]} and {results[1].split(' = ')[-1]}"
        )
    else:
        # For more operations, use " | "
        final_result = " | ".join(results)

    msgs.append(AIMessage(content=intermediate_response))
    msgs.append(AIMessage(content=final_result))
    new_state["messages"] = msgs

    return new_state


def sequential_executor_node(state: ChatState) -> ChatState:
    """Executes ONE sequential step and updates the state."""
    steps = state.get("steps") or []
    current_step = state.get("current_step") or 0
    results = state.get("results") or []

    if current_step >= len(steps):
        # No more steps, terminate
        return state

    step = steps[current_step]
    result = ""
    step_description = ""

    if step["type"] == "calc":
        step_description = f"Calculation: {step['operation']}"
        result = tool_calculator(step["operation"])
    elif step["type"] == "time":
        step_description = "Time query"
        result = tool_time("time")
    elif step["type"] == "rag":
        step_description = f"Search: {step['query']}"
        docs = mock_retrieve(step["query"])
        result = mock_rag_answer(step["query"], docs)
    elif step["type"] == "sum":
        step_description = "Sum of previous results"
        # Sum the results from previous calculations
        calc_results = [r for r in results if "Result (mock calc):" in r]
        if calc_results:
            # Extract numbers from results
            numbers = []
            for calc_result in calc_results:
                # Find the number after "="
                match = re.search(r"=\s*(\d+)", calc_result)
                if match:
                    numbers.append(int(match.group(1)))

            if numbers:
                total = sum(numbers)
                result = f"Total sum: {' + '.join(map(str, numbers))} = {total}"
            else:
                result = "Could not sum the results"
        else:
            result = "No calculation results to sum"

    # Create structured response
    new_state = state.copy()
    msgs = list(new_state["messages"])

    # Show current step - only show header for first step
    if current_step == 0:
        step_response = f"""⏭️  Executing operations sequentially...

  {(current_step or 0) + 1}. {step_description} → {result}"""
    else:
        step_response = f"  {(current_step or 0) + 1}. {step_description} → {result}"

    msgs.append(AIMessage(content=step_response))
    new_state["messages"] = msgs

    # Update state for next step
    new_state["current_step"] = (current_step or 0) + 1
    # Don't use Annotated here, just add the current result
    new_state["results"] = results + [result]

    return new_state


def finish_node(state: ChatState) -> ChatState:
    """Finish node."""
    # In a real case, the supervisor could post-process/validate/combine.
    return state
