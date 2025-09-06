"""
Main CLI for the multi-agent system.
"""

import sys

from .chat import run_chat


def main():
    """Main CLI function."""
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            # Test mode - run integration tests
            import subprocess

            result = subprocess.run(
                ["uv", "run", "pytest", "tests/integration/", "-v"],
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            sys.exit(result.returncode)
        elif sys.argv[1] == "--debug":
            # Chat mode with debug
            run_chat(debug_mode=True)
        elif sys.argv[1] == "--historial":
            # Chat mode with history
            run_chat(history_mode=True)
        elif sys.argv[1] in ["--help", "-h"]:
            # Show help
            print("🤖 MULTI-AGENT CHATBOT - Available options:")
            print()
            print("  uv run multi-agent                 # Normal interactive chat")
            print(
                "  uv run multi-agent --debug         # Chat with complete state object"
            )
            print(
                "  uv run multi-agent --historial     # Chat with conversation history"
            )
            print("  uv run multi-agent --test          # Run tests")
            print("  uv run multi-agent --help          # Show this help")
            print()
        else:
            print(f"❌ Unknown option: {sys.argv[1]}")
            print("💡 Use --help to see available options")
            sys.exit(1)
    else:
        # Normal interactive chat mode
        run_chat()


if __name__ == "__main__":
    main()
