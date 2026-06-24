"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


import logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.core.config import get_settings


logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        max_iterations = self.settings.max_iterations

        # Guardrail: Check maximum iteration limit
        if state.iteration >= max_iterations:
            logger.warning(f"Max iterations reached ({state.iteration}/{max_iterations}). Routing to done.")
            state.record_route("done")
            return state

        # Determine next step using LLM
        system_prompt = (
            "You are the Supervisor of a multi-agent research team. Your job is to orchestrate the research process by calling the right agent next or ending the process.\n"
            "The available agents are:\n"
            "- researcher: Gathers sources and drafts raw research notes.\n"
            "- analyst: Processes research notes to extract claims, viewpoints, and inconsistencies.\n"
            "- writer: Synthesizes research and analysis notes into a final cited answer.\n"
            "- critic: Reviews the final answer for citation correctness, completeness, and hallucinations.\n"
            "- done: Select this when the final answer has been written and approved by the critic, or if no further improvements can be made.\n\n"
            "Routing Rules:\n"
            "1. If no research notes exist, call 'researcher'.\n"
            "2. If research notes exist but no analysis notes, call 'analyst'.\n"
            "3. If analysis notes exist but no draft final answer, call 'writer'.\n"
            "4. If a draft final answer exists but has not been reviewed by the critic, call 'critic'.\n"
            "5. If the critic has run and output '[APPROVED]', call 'done'.\n"
            "6. If the critic has run and output '[REJECTED]' with feedback, you can call 'researcher' (to get more info), 'analyst' (to re-analyze), or 'writer' (to re-write) depending on what is needed. If you are close to the maximum iteration limit, select 'done' directly.\n\n"
            "You MUST output exactly one word in lowercase: researcher, analyst, writer, critic, or done."
        )

        # Assemble summary of what has been done so far
        history_summary = (
            f"Current Iteration: {state.iteration}\n"
            f"Route History: {state.route_history}\n"
            f"Sources Gathered: {len(state.sources)}\n"
            f"Has Research Notes: {state.research_notes is not None}\n"
            f"Has Analysis Notes: {state.analysis_notes is not None}\n"
            f"Has Draft Final Answer: {state.final_answer is not None}\n"
        )
        if state.agent_results:
            last_result = state.agent_results[-1]
            history_summary += f"Last Agent Active: {last_result.agent}\n"
            if last_result.agent == "critic":
                history_summary += f"Critic Result Summary: {last_result.content[:300]}...\n"

        user_prompt = f"Workflow State:\n{history_summary}\nDecide the next action:"

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            decision = response.content.strip().lower()
            # Clean up the output in case the LLM wrapped it in punctuation or markdown
            decision = "".join(char for char in decision if char.isalnum() or char in ["_", "-"])

            valid_routes = {"researcher", "analyst", "writer", "critic", "done"}
            if decision not in valid_routes:
                logger.warning(f"LLM returned invalid routing decision: {decision}. Falling back to default routing.")
                decision = self._deterministic_fallback(state)
        except Exception as exc:
            logger.error(f"Supervisor LLM call failed: {exc}. Falling back to deterministic routing.")
            decision = self._deterministic_fallback(state)

        logger.info(f"Supervisor decided to route to: {decision}")
        state.record_route(decision)
        return state

    def _deterministic_fallback(self, state: ResearchState) -> str:
        """Deterministic routing fallback when LLM fails or returns invalid values."""
        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        # If critic has not run yet in this iteration loop
        last_agents = [res.agent for res in state.agent_results]
        if "critic" not in last_agents or last_agents[-1] != "critic":
            return "critic"
        return "done"
