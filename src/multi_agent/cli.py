"""
Main CLI for the multi-agent API debugger system.
"""

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
    run_chat(debug_mode=debug, history_mode=history, thread_id=thread_id)


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
    from .demo import run_demo

    run_demo(thread_id=thread_id)


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
    click.echo()
    click.echo("💡 EXAMPLE COMMANDS:")
    click.echo("  • 'list all jobs' - Show available jobs")
    click.echo("  • 'run job data_processing' - Execute a job")
    click.echo("  • 'debug job_003' - Analyze job failure")
    click.echo("  • 'what is the API?' - Get system information")
    click.echo()
    click.echo("🚀 GET STARTED:")
    click.echo("  multi-agent chat")
    click.echo()


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
