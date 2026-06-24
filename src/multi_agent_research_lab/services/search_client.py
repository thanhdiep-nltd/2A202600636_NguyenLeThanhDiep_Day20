"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument


import json
import logging
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client that simulates search results using LLM."""

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        This uses the LLM to generate realistic search snippets dynamically,
        making the mock search adaptable to any topic.
        """
        logger.info(f"Simulating web search for query: {query}")

        system_prompt = (
            "You are a Mock Web Search Engine. Your task is to simulate realistic and informative web search results for the user's query.\n"
            "Generate high-quality, factual-looking snippets containing realistic details, data points, or state-of-the-art information.\n"
            "You MUST return a JSON array containing object elements matching this JSON schema:\n"
            "[\n"
            "  {\n"
            "    \"title\": \"Title of the web page\",\n"
            "    \"url\": \"https://example.com/some-path\",\n"
            "    \"snippet\": \"A relevant, informative snippet containing facts, numbers, and useful information for the query.\"\n"
            "  }\n"
            "]\n"
            "Return only valid JSON. Do not include markdown code block formatting (like ```json) in your response. "
            f"Provide exactly {max_results} results."
        )

        user_prompt = f"Perform search query: {query}"

        try:
            response = self.llm_client.complete(system_prompt, user_prompt)
            raw_content = response.content.strip()

            # Strip out markdown backticks if the model outputs them
            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()

            results_data = json.loads(raw_content)
            documents = []
            for item in results_data[:max_results]:
                documents.append(
                    SourceDocument(
                        title=item.get("title", "Untitled Source"),
                        url=item.get("url", "https://example.com"),
                        snippet=item.get("snippet", ""),
                        metadata={"simulated": True}
                    )
                )
            return documents
        except Exception as exc:
            logger.warning(f"Failed to generate dynamic search results: {exc}. Falling back to static mock.")
            # Fallback static mock
            return [
                SourceDocument(
                    title=f"Introduction to {query}",
                    url="https://example.com/intro",
                    snippet=f"This is a fallback mock document containing basic information about {query} to ensure the application continues running smoothly.",
                    metadata={"fallback": True}
                )
            ]
