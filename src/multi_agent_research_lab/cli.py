from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery, AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.services.storage import LocalArtifactStore

import os
from dotenv import load_dotenv

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    load_dotenv()
    settings = get_settings()
    configure_logging(settings.log_level)
    
    # Expose LangSmith configs to os.environ so LangGraph auto-tracing is triggered
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"


def _run_baseline(query: str) -> ResearchState:
    """Helper runner to execute single-agent baseline."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    llm = LLMClient()

    system_prompt = (
        "You are an expert research assistant. Answer the user query comprehensively, "
        "relying on your internal knowledge. Structure your output clearly using markdown. "
        "Do not use external search or agents."
    )
    response = llm.complete(system_prompt, query)

    state.final_answer = response.content
    metadata = {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
    }
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata=metadata
        )
    )
    return state


def _run_multi_agent(query: str) -> ResearchState:
    """Helper runner to execute multi-agent workflow."""
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a real single-agent baseline implementation."""
    _init()
    console.print(f"[bold blue]Running Single-Agent Baseline for query:[/bold blue] {query}")
    state = _run_baseline(query)
    console.print(Panel(state.final_answer or "No response generated.", title="Single-Agent Baseline Result"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    console.print(f"[bold green]Running Multi-Agent Workflow for query:[/bold green] {query}")
    try:
        result = _run_multi_agent(query)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    console.print(Panel(result.final_answer or "No response generated.", title="Multi-Agent Result"))
    console.print("[bold cyan]Agent Execution Order:[/bold cyan]", " -> ".join(result.route_history))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run single-agent vs multi-agent back-to-back and output markdown report."""
    _init()
    console.print(f"[bold magenta]Starting Benchmark Comparison for query:[/bold magenta] {query}\n")

    # 1. Run Baseline
    console.print("[yellow]Executing Single-Agent Baseline...[/yellow]")
    baseline_state, baseline_metrics = run_benchmark("baseline", query, _run_baseline)

    # 2. Run Multi-Agent
    console.print("[yellow]Executing Multi-Agent Workflow...[/yellow]")
    multi_state, multi_metrics = run_benchmark("multi-agent", query, _run_multi_agent)

    # 3. Render report
    metrics_list = [baseline_metrics, multi_metrics]
    report_content = render_markdown_report(metrics_list)

    # 4. Save to reports/benchmark_report.md
    store = LocalArtifactStore()
    report_path = store.write_text("benchmark_report.md", report_content)
    console.print(f"\n[green]Benchmark report successfully written to:[/green] {report_path.resolve()}\n")

    # 5. Display comparative table in console
    table = Table(title="Benchmark Comparison Summary")
    table.add_column("Run Name", style="cyan")
    table.add_column("Latency", style="magenta")
    table.add_column("Cost (USD)", style="green")
    table.add_column("Quality Score", style="bold yellow")
    table.add_column("Notes", style="white")

    for metric in metrics_list:
        cost = "N/A" if metric.estimated_cost_usd is None else f"${metric.estimated_cost_usd:.6f}"
        quality = "N/A" if metric.quality_score is None else f"{metric.quality_score:.1f}/10.0"
        table.add_row(
            metric.run_name,
            f"{metric.latency_seconds:.2f}s",
            cost,
            quality,
            metric.notes
        )
    console.print(table)


if __name__ == "__main__":
    app()
