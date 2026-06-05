from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    def chat(self, messages: list[dict], model: str, **kwargs) -> str:
        """Send messages to the LLM and return the text reply."""
        ...


@runtime_checkable
class EmbedProvider(Protocol):
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed a batch of texts and return their vectors."""
        ...
