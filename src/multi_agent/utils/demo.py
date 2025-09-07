"""
Demo module for showcasing the multi-agent API debugger capabilities.
"""

from langchain_core.messages import HumanMessage

from ..graph import get_graph


async def run_demo(thread_id: str = "demo"):
    """Run a demonstration of the system capabilities."""
    print("🎬 MULTI-AGENT API DEBUGGER DEMO")
    print("=" * 50)
    print()

    # Demo scenarios
    scenarios = [
        {
            "title": "1. API Operations - List Jobs",
            "command": "list all jobs",
            "description": "Demonstrates the API Operator listing available jobs",
        },
        {
            "title": "2. Knowledge Query - API Information",
            "command": "what is the API?",
            "description": "Shows the Knowledge Assistant answering questions",
        },
        {
            "title": "3. Error Debugging - Job Failure Analysis",
            "command": "debug the error in job_003",
            "description": "Demonstrates the Debugger analyzing job failures",
        },
        {
            "title": "4. API Operations - Run Job",
            "command": "run job data_processing",
            "description": "Shows executing a job through the API Operator",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"🎯 {scenario['title']}")
        print(f"   Command: '{scenario['command']}'")
        print(f"   Description: {scenario['description']}")
        print()

        # Execute the command
        print("🤖 Processing...")
        graph = get_graph()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=scenario["command"])]},
            config={"configurable": {"thread_id": f"{thread_id}-{i}"}},
        )

        # Show the response
        if result["messages"]:
            # Get the last few messages (excluding the user input)
            ai_messages = [
                msg
                for msg in result["messages"][1:]
                if hasattr(msg, "type") and msg.type == "ai"
            ]

            for msg in ai_messages:
                print(msg.content)

        print()
        print("-" * 50)
        print()

        # Pause between scenarios
        if i < len(scenarios):
            user_input = (
                input(
                    "Press Enter to continue, 'q' to quit, or 's' to skip remaining: "
                )
                .strip()
                .lower()
            )
            if user_input == "q":
                print("\n👋 Demo stopped by user.")
                break
            elif user_input == "s":
                print("\n⏭️  Skipping remaining scenarios...")
                break
            print()

    print("🎉 Demo completed! The system successfully demonstrated:")
    print("   ✅ API Operations (list jobs, run jobs)")
    print("   ✅ Knowledge Base queries")
    print("   ✅ Error debugging and root cause analysis")
    print("   ✅ Response synthesis and formatting")
    print()
    print("🚀 Try the interactive chat: multi-agent-api-debugger chat")
