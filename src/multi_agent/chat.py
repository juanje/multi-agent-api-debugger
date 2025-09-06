"""
Interactive chat for the multi-agent system.
"""

from langchain_core.messages import HumanMessage

from .graph import app


def print_welcome():
    """Shows the welcome message and system capabilities."""
    print("=" * 60)
    print("🤖 MULTI-AGENT CHATBOT (LangGraph 0.3.x)")
    print("=" * 60)
    print()
    print("📋 AVAILABLE TOOLS:")
    print("  🔧 CALCULATOR: Mathematical operations (e.g: 12*7, 5+3, 10/2)")
    print("  ⏰ TIME: Query current time (e.g: what time is it?)")
    print(
        "  📚 RAG: Document search (e.g: summarize the manual, search for information)"
    )
    print()
    print("💡 TYPES OF QUESTIONS YOU CAN ASK:")
    print("  • Calculations: 'What is 15*8?', 'calculate 100-25'")
    print("  • Time: 'What time is it?', 'tell me the current time'")
    print("  • Documents: 'summarize the manual', 'search for information about X'")
    print(
        "  • PARALLEL: 'What is 2+3 and 8/2?', 'tell me the time and search the manual'"
    )
    print(
        "  • SEQUENTIAL: 'sum of 3x8 and 129/3', 'search for information about tasks for this time'"
    )
    print("  • General questions: 'can you help me?' (goes to RAG)")
    print()
    print("🚪 To exit type: 'bye' or 'exit'")
    print("=" * 60)
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


def run_chat(debug_mode: bool = False, history_mode: bool = False):
    """Runs the interactive chat."""
    print_welcome()

    if debug_mode:
        print("🐛 [DEBUG] Debug mode activated - complete state object will be shown")
        print()
    elif history_mode:
        print("📜 [HISTORY] History mode activated - chat history will be shown")
        print()

    print("💬 Hello! I'm your multi-agent assistant. How can I help you?")
    print()

    # Use a fixed thread_id for the entire chat session
    thread_id = "chat_session"

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
                "route": None,
                "operations": None,
                "steps": None,
                "results": None,
                "current_step": None,
                "pending_tasks": None,
            }

            # Execute the graph
            for _ in app.stream(
                state, config={"configurable": {"thread_id": thread_id}}
            ):
                pass

            # Get final result
            final_state = app.get_state(
                {"configurable": {"thread_id": thread_id}}
            ).values

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

                # Don't show route for any operation
                # (route display logic was removed)

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
