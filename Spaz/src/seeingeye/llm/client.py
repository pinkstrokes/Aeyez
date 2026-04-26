"""LLMClient -- holds pre-configured ChatOpenAI instances for both agents.

This is the single entry-point for obtaining LLM clients in the seeingeye
package.  It intentionally has zero imports from ``agents``, ``graph``,
``tools``, or ``state`` to remain a pure leaf dependency.
"""

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from .vllm_openai import create_reasoner_client, create_translator_client


@dataclass
class LLMClient:
    """Pre-configured ChatOpenAI instances for the Translator and Reasoner.

    Use :meth:`from_defaults` for the standard paper configuration
    (Translator on port 8000, Reasoner on port 8001).
    """

    translator: ChatOpenAI
    reasoner: ChatOpenAI

    @classmethod
    def from_defaults(cls) -> "LLMClient":
        """Construct with paper-default endpoints and sampling parameters."""
        return cls(
            translator=create_translator_client(),
            reasoner=create_reasoner_client(),
        )
