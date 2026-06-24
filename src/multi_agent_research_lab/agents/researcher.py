"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.core.schemas import AgentName, AgentResult


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self.search_client = SearchClient()
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        # 1. Search for sources
        sources = self.search_client.search(query, max_results=max_sources)
        state.sources = sources

        if not sources:
            state.research_notes = "No relevant sources found."
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content=state.research_notes
                )
            )
            return state

        # 2. Formulate summary prompt
        system_prompt = (
            "You are an expert Research Agent. Your task is to review the raw search results and compile structured, "
            "concise research notes that highlight the key findings, data, and details relevant to the query.\n"
            "Keep the notes factual, detailed, and directly cite the sources by their index (e.g. [0], [1])."
        )

        user_prompt = f"Query: {query}\n\nSearch Results:\n"
        for idx, doc in enumerate(sources):
            user_prompt += f"Source [{idx}]:\nTitle: {doc.title}\nURL: {doc.url}\nSnippet: {doc.snippet}\n\n"

        # 3. Call LLM
        response = self.llm_client.complete(system_prompt, user_prompt)

        # 4. Save updates to state
        state.research_notes = response.content
        metadata = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        }
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata=metadata
            )
        )

        return state
