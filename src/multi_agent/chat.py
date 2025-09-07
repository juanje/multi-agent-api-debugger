"""
Interactive chat for the multi-agent API debugger system.
"""

import os
import logging
from langchain_core.messages import HumanMessage
from .graph import get_graph

# Disable LLM mocking to use real LLMs
os.environ["USE_LLM_MOCKS"] = "false"

# Configure logging to show only important messages
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Set specific loggers to INFO for debugging
logging.getLogger("multi_agent.supervisor").setLevel(logging.INFO)
logging.getLogger("multi_agent.routing").setLevel(logging.INFO)


def print_welcome():
    """Shows the welcome message and system capabilities."""
    print("=" * 70)
    print("🤖 MULTI-AGENT API DEBUGGER")
    print("=" * 70)
    print()
    print("📋 SYSTEM CAPABILITIES:")
    print("  🔧 API OPERATIONS: Run jobs, get results, check system status")
    print("  🐞 ERROR DEBUGGING: Analyze failures and provide root cause analysis")
    print("  📚 KNOWLEDGE BASE: Answer questions about the system and API")
    print("  📝 RESPONSE SYNTHESIS: Format and present results clearly")
    print()
    print("💡 EXAMPLE COMMANDS:")
    print("  • API Operations:")
    print("    - 'list all jobs' - Show available jobs")
    print("    - 'run job data_processing' - Execute a specific job")
    print("    - 'get job results job_001' - Get results for a job")
    print("    - 'check system status' - Monitor system health")
    print()
    print("  • Debugging:")
    print("    - 'debug job_003' - Analyze why job_003 failed")
    print("    - 'investigate the error' - Debug the last error")
    print()
    print("  • Knowledge Queries:")
    print("    - 'what is the API?' - Learn about the API")
    print("    - 'how do I authenticate?' - Get authentication help")
    print("    - 'explain job templates' - Learn about templates")
    print()
    print("🚪 To exit type: 'bye', 'exit', or 'quit'")
    print("=" * 70)
    print()


def print_chat_history(state: dict, route: str):
    """Prints the chat history."""
    print("📜 [HISTORY] Conversation:")
    print(f"   📍 Route: {route}")
    print(f"   💬 Messages: {len(state.get('messages', []))}")
    for i, msg in enumerate(state.get("messages", [])):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        print(
            f"      {i + 1}. {role}: {content[:50]}{'...' if len(str(content)) > 50 else ''}"
        )
    print()


def print_debug_state(state: dict, route: str):
    """Prints the complete state object of the graph for debug."""
    print("🔍 [DEBUG] Complete state object:")
    print(f"   📍 Route: {route}")
    print("   📊 Complete state:")
    print(f"   {state}")
    print()


async def run_chat(
    debug_mode: bool = False, history_mode: bool = False, thread_id: str = "default"
):
    """Runs the interactive chat."""
    print_welcome()

    # LLM service will be initialized on demand
    print("🔄 LLM service will be initialized on demand...")
    print("✅ Ready to start")
    print()

    if debug_mode:
        print("🐛 [DEBUG] Debug mode activated - complete state object will be shown")
        print()
    elif history_mode:
        print("📜 [HISTORY] History mode activated - chat history will be shown")
        print()

    print("💬 Hello! I'm your multi-agent API debugger. How can I help you?")
    print()

    # Counter to track assistant messages shown
    ai_messages_shown = 0

    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()

            # Check exit command
            if user_input.lower() in ["bye", "exit", "quit", "goodbye"]:
                print("\n🤖 Assistant: Goodbye! It was a pleasure helping you. 👋")
                break

            # Check empty input
            if not user_input:
                print("🤖 Assistant: Please write something so I can help you.")
                continue

            # Process with the graph
            print("🤖 Assistant: Processing...")

            # Create initial state with user message
            state = {
                "messages": [HumanMessage(content=user_input)],
                "goal": user_input,
                "todo_list": None,
                "results": None,
                "error_info": None,
                "root_cause_analysis": None,
                "final_response": None,
                "route": None,
                "next_agent": None,
                "knowledge_summary": None,
            }

            # Execute the graph asynchronously
            graph = get_graph()
            final_state = await graph.ainvoke(
                state, config={"configurable": {"thread_id": thread_id}}
            )

            # Show response
            if final_state["messages"]:
                route = final_state.get("route", "unknown")

                # Get only assistant messages from this conversation
                ai_messages = [
                    msg
                    for msg in final_state["messages"]
                    if hasattr(msg, "type") and msg.type == "ai"
                ]

                # Show only new assistant messages
                new_ai_messages = ai_messages[ai_messages_shown:]

                if new_ai_messages:
                    # Show all intermediate messages without prefix
                    for i, msg in enumerate(new_ai_messages[:-1]):
                        print(msg.content)

                    # Add blank line before final result
                    print()

                    # Show final message with prefix (only if it doesn't already have the prefix)
                    if len(new_ai_messages) > 0:
                        final_msg = new_ai_messages[-1].content
                        if final_msg.startswith("🤖 Assistant: "):
                            print(final_msg)
                        else:
                            print(f"🤖 Assistant: {final_msg}")

                    # Update counter
                    ai_messages_shown = len(ai_messages)

                # Show debug or history if activated
                if debug_mode:
                    print_debug_state(final_state, route)
                elif history_mode:
                    print_chat_history(final_state, route)
            else:
                print("🤖 Assistant: Sorry, I couldn't process your request.")

            print()

        except KeyboardInterrupt:
            print("\n\n🤖 Assistant: Goodbye! It was a pleasure helping you. 👋")
            break
        except Exception as e:
            print(f"🤖 Assistant: Oops, an error occurred: {e}")
            print("Please try again.")
            print()
