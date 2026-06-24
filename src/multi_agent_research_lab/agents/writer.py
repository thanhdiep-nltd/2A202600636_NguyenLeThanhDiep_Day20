"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.core.schemas import AgentName, AgentResult


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        query = state.request.query
        audience = state.request.audience
        research_notes = state.research_notes or "No research notes available."
        analysis_notes = state.analysis_notes or "No analysis notes available."

        sources_text = ""
        for idx, doc in enumerate(state.sources):
            sources_text += f"[{idx}] {doc.title} ({doc.url or 'No URL'})\n"

        system_prompt = (
            f"You are an expert Technical Writer Agent. Your task is to write a comprehensive, clear, and well-structured final answer "
            f"for the user's query. Tailor your response to the target audience: '{audience}'.\n"
            f"Synthesize the provided research notes and analysis notes into a cohesive, high-quality, professional report.\n\n"
            f"Requirements:\n"
            f"- Use Markdown formatting (headings, bullet points, code snippets, etc.).\n"
            f"- You MUST include inline citations (e.g. [0], [1]) pointing to the source indices.\n"
            f"- End the report with a 'References' section listing all cited sources (showing title and URL)."
        )

        user_prompt = (
            f"User Query: {query}\n"
            f"Target Audience: {audience}\n\n"
            f"Research Notes:\n{research_notes}\n\n"
            f"Analysis Notes:\n{analysis_notes}\n\n"
            f"Available Sources:\n{sources_text}"
        )

        response = self.llm_client.complete(system_prompt, user_prompt)

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
