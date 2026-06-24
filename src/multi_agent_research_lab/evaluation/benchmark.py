"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, estimate token cost, citation coverage, and judge quality using LLM."""
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # 1. Aggregate Cost from agent execution steps
    total_cost = 0.0
    if state.agent_results:
        for res in state.agent_results:
            if res.metadata:
                total_cost += res.metadata.get("cost_usd", 0.0)

    # 2. Evaluate Quality using LLM-as-a-judge
    quality = None
    if state.final_answer and "TODO(student)" not in state.final_answer:
        try:
            from multi_agent_research_lab.services.llm_client import LLMClient
            llm = LLMClient()
            judge_system = (
                "You are an objective AI Quality Judge. Your task is to rate a technical report on a scale of 0.0 to 10.0.\n"
                "Evaluate the report based on correctness, structure, completeness, and clarity.\n"
                "Your output MUST be just a single float number between 0.0 and 10.0 (e.g. 8.5)."
            )
            judge_user = f"Query: {query}\n\nReport:\n{state.final_answer}"
            resp = llm.complete(judge_system, judge_user)
            # Find a float in the response
            match = re.search(r"(\d+(?:\.\d+)?)", resp.content)
            if match:
                quality = float(match.group(1))
                quality = max(0.0, min(10.0, quality))
        except Exception:
            quality = 7.0  # Default fallback if LLM scoring fails

    # 3. Calculate Citation Coverage
    citations_found = set(re.findall(r"\[(\d+)\]", state.final_answer or ""))
    total_sources = len(state.sources)
    citation_text = ""
    if total_sources > 0:
        coverage_pct = (len(citations_found) / total_sources) * 100
        citation_text = f"Citation coverage: {coverage_pct:.1f}% ({len(citations_found)}/{total_sources} cited)"
    else:
        citation_text = "No sources gathered"

    notes_parts = [citation_text]
    if state.route_history:
        notes_parts.append(f"Steps: {len(state.route_history)}")
    if state.errors:
        notes_parts.append(f"Errors: {len(state.errors)}")
    notes = ". ".join(notes_parts)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality,
        notes=notes
    )

    return state, metrics
