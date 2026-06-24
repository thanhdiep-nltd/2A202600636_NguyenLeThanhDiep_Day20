"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.core.schemas import AgentName, AgentResult


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        notes = state.research_notes or "No research notes available."

        system_prompt = (
            "You are a Senior Analyst Agent. Your task is to critique and analyze the compiled research notes.\n"
            "Analyze the information systematically:\n"
            "1. Extract the core claims and factual assertions.\n"
            "2. Compare conflicting viewpoints or approaches if any exist.\n"
            "3. Identify any gaps in the research, questionable evidence, or areas requiring further investigation.\n"
            "4. Organize your findings into clear structured analysis notes with bullet points."
        )

        user_prompt = f"Research Notes to Analyze:\n{notes}"

        response = self.llm_client.complete(system_prompt, user_prompt)

        state.analysis_notes = response.content
        metadata = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        }
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata=metadata
            )
        )

        return state
