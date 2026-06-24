from unittest.mock import MagicMock, patch

from multi_agent_research_lab.agents import SupervisorAgent, ResearcherAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


@patch("multi_agent_research_lab.services.llm_client.OpenAI")
def test_supervisor_agent_routing(mock_openai) -> None:
    # Setup mock LLM response returning the next route
    mock_instance = mock_openai.return_value
    mock_chat = MagicMock()
    mock_instance.chat = mock_chat
    mock_completion = MagicMock()
    mock_chat.completions.create.return_value = mock_completion
    mock_completion.choices = [MagicMock(message=MagicMock(content="researcher"))]
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    agent = SupervisorAgent()
    res = agent.run(state)
    assert res.iteration == 1
    assert "researcher" in res.route_history


@patch("multi_agent_research_lab.services.search_client.SearchClient.search")
@patch("multi_agent_research_lab.services.llm_client.OpenAI")
def test_researcher_agent_execution(mock_openai, mock_search) -> None:
    # Setup mock search and mock LLM response
    mock_search.return_value = [
        SourceDocument(title="Test Source", snippet="Interesting info", url="http://test.com")
    ]
    mock_instance = mock_openai.return_value
    mock_chat = MagicMock()
    mock_instance.chat = mock_chat
    mock_completion = MagicMock()
    mock_chat.completions.create.return_value = mock_completion
    mock_completion.choices = [MagicMock(message=MagicMock(content="Researcher notes summary"))]
    mock_completion.usage = MagicMock(prompt_tokens=15, completion_tokens=10)

    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    agent = ResearcherAgent()
    res = agent.run(state)
    assert len(res.sources) == 1
    assert res.research_notes == "Researcher notes summary"
    assert len(res.agent_results) == 1
