"""
Main CLI for the multi-agent API debugger system.
"""

import asyncio
import subprocess
import sys

import click

from .chat import run_chat


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Multi-Agent API Debugger - Intelligent system for job management and debugging."""
    pass


@cli.command()
@click.option("--debug", is_flag=True, help="Show complete state object for debugging")
@click.option("--history", is_flag=True, help="Show conversation history")
@click.option(
    "--thread-id", default="default", help="Thread ID for conversation persistence"
)
def chat(debug: bool, history: bool, thread_id: str):
    """Start interactive chat with the multi-agent system."""
    asyncio.run(run_chat(debug_mode=debug, history_mode=history, thread_id=thread_id))


@cli.command()
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--coverage", is_flag=True, help="Run with coverage report")
def test(verbose: bool, coverage: bool):
    """Run the test suite."""
    cmd = ["uv", "run", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html"])

    cmd.append("tests/")

    result = subprocess.run(cmd, capture_output=True, text=True)

    click.echo(result.stdout)
    if result.stderr:
        click.echo(result.stderr, err=True)

    sys.exit(result.returncode)


@cli.command()
@click.option("--thread-id", default="demo", help="Thread ID for the demo")
def demo(thread_id: str):
    """Run a demonstration of the system capabilities."""
    import asyncio
    from .utils.demo import run_demo

    asyncio.run(run_demo(thread_id=thread_id))


@cli.command()
@click.option("--stats", is_flag=True, help="Show memory statistics")
@click.option("--recent", type=int, default=5, help="Show recent memories (default: 5)")
def memory(stats: bool, recent: int):
    """Manage and view Long Term Memory (LTM) information."""

    async def show_memory_info():
        try:
            from .memory import get_ltm_service

            ltm_service = get_ltm_service()

            if stats:
                click.echo("💾 Long Term Memory Statistics")
                click.echo("=" * 40)

                memory_stats = await ltm_service.get_stats()
                click.echo(
                    f"📊 Total memories stored: {memory_stats['total_memories']}"
                )
                click.echo(
                    f"🔧 Vector store initialized: {memory_stats['vector_store_initialized']}"
                )

                if memory_stats["total_memories"] > 0:
                    click.echo()
                    click.echo("📝 Recent interactions:")
                    recent_memories = await ltm_service.get_recent_interactions(
                        limit=recent
                    )

                    for i, memory in enumerate(recent_memories, 1):
                        memory_type_icon = {
                            "qa_session": "💬",
                            "debug_analysis": "🐞",
                            "api_operation": "🔧",
                            "knowledge_query": "📚",
                        }.get(memory.type.value, "❓")

                        query_preview = (
                            memory.user_query[:50] + "..."
                            if len(memory.user_query) > 50
                            else memory.user_query
                        )
                        click.echo(
                            f"  {i}. {memory_type_icon} {query_preview} ({memory.timestamp[:10]})"
                        )
                else:
                    click.echo()
                    click.echo(
                        "📝 No memories stored yet. Start chatting to build your memory!"
                    )

            else:
                click.echo("💾 Long Term Memory (LTM)")
                click.echo("=" * 30)
                click.echo()
                click.echo("📋 WHAT IS LTM?")
                click.echo(
                    "  LTM stores your conversations and system interactions to make"
                )
                click.echo("  future responses more intelligent and context-aware.")
                click.echo()
                click.echo("🧠 WHAT IS STORED:")
                click.echo("  💬 Q&A sessions - Your questions and system responses")
                click.echo("  🐞 Debug analyses - Error investigations and solutions")
                click.echo("  🔧 API operations - Successful and failed API calls")
                click.echo("  📚 Knowledge queries - Information searches and answers")
                click.echo()
                click.echo("🚀 HOW IT HELPS:")
                click.echo("  • Debugger finds similar past errors for better analysis")
                click.echo("  • Knowledge assistant references previous conversations")
                click.echo("  • System learns patterns from your interactions")
                click.echo()
                click.echo("💡 COMMANDS:")
                click.echo("  multi-agent memory --stats    # Show memory statistics")
                click.echo(
                    "  multi-agent memory --recent 10  # Show 10 recent memories"
                )
                click.echo()

        except ImportError:
            click.echo("❌ LTM dependencies not installed. Run: uv sync")
        except Exception as e:
            click.echo(f"❌ Error accessing LTM: {e}")

    asyncio.run(show_memory_info())


@cli.command()
def info():
    """Show system information and capabilities."""
    click.echo("🤖 Multi-Agent API Debugger")
    click.echo("=" * 40)
    click.echo()
    click.echo("📋 CAPABILITIES:")
    click.echo("  🔧 API Operations: Run jobs, get results, check status")
    click.echo("  🐞 Error Debugging: Root cause analysis and troubleshooting")
    click.echo("  📚 Knowledge Base: Answer questions about the system")
    click.echo("  📝 Response Synthesis: Format and present results")
    click.echo("  💾 Long Term Memory: Learn from past interactions")
    click.echo()
    click.echo("💡 EXAMPLE COMMANDS:")
    click.echo("  • 'list all jobs' - Show available jobs")
    click.echo("  • 'run job data_processing' - Execute a job")
    click.echo("  • 'debug job_003' - Analyze job failure")
    click.echo("  • 'what is the API?' - Get system information")
    click.echo()
    click.echo("🚀 GET STARTED:")
    click.echo("  multi-agent chat")
    click.echo("  multi-agent memory --stats")
    click.echo()


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
