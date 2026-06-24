"""LangGraph workflow skeleton."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from langgraph.graph import StateGraph, END
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.agents import (
    SupervisorAgent,
    ResearcherAgent,
    AnalystAgent,
    WriterAgent,
    CriticAgent,
)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()
        self.graph = self.build()

    def build(self) -> object:
        """Create a LangGraph graph."""
        builder = StateGraph(ResearchState)

        # 1. Add agent nodes
        builder.add_node("supervisor", self.supervisor.run)
        builder.add_node("researcher", self.researcher.run)
        builder.add_node("analyst", self.analyst.run)
        builder.add_node("writer", self.writer.run)
        builder.add_node("critic", self.critic.run)

        # 2. Set entrypoint
        builder.set_entry_point("supervisor")

        # 3. Add edges from workers back to supervisor
        builder.add_edge("researcher", "supervisor")
        builder.add_edge("analyst", "supervisor")
        builder.add_edge("writer", "supervisor")
        builder.add_edge("critic", "supervisor")

        # 4. Add conditional routing from supervisor
        def route_next(state: ResearchState) -> str:
            if not state.route_history:
                return END
            next_agent = state.route_history[-1]
            if next_agent == "done":
                return END
            return next_agent

        builder.add_conditional_edges(
            "supervisor",
            route_next,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END
            }
        )

        return builder.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        # The compile method returns a compiled graph which supports invoke()
        result = self.graph.invoke(state)
        if isinstance(result, dict):
            return ResearchState.model_validate(result)
        return result
