"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from multi_agent_research_lab.core.errors import StudentTodoError


from openai import OpenAI
from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client implementation."""

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_api_base
        self.model = settings.openai_model
        self.timeout = float(settings.timeout_seconds)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=3,
            timeout=self.timeout
        )

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        # Prices per 1M tokens for gpt-4o-mini and gpt-4o
        prices = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
        }
        # Fallback to gpt-4o-mini pricing
        in_rate, out_rate = prices.get(model.lower(), (0.15, 0.60))
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost_usd = self._estimate_cost(self.model, input_tokens, output_tokens)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
