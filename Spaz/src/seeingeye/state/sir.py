"""Structured Intermediate Representation (SIR) — the load-bearing data model
connecting Translator and Reasoner in the SeeingEye agentic flow.

Wraps a plain string with typed operations replicating the exact merge
semantics from ``iterative_refinement.py``. All methods return NEW instances
-- never mutate in place.
"""

from __future__ import annotations

from pydantic import BaseModel


class SIR(BaseModel):
    """Structured Intermediate Representation.

    A Pydantic v2 model wrapping a plain string (``content``) with three
    mutation operations that mirror the original codebase:

    * :meth:`merge_feedback` -- append reasoning feedback (S_i <- S_i + F)
    * :meth:`update` -- incremental update within the translator inner loop
    * :meth:`replace` -- full content replacement

    Every method returns a **new** ``SIR`` instance; the original is never
    mutated.
    """

    content: str = ""

    def merge_feedback(self, feedback: str) -> SIR:
        """Append reasoning feedback with ``--- REASONING FEEDBACK ---`` separator.

        Replicates ``_append_feedback_to_sir()`` from
        ``iterative_refinement.py`` (lines 83-102).

        Guards:
        * Empty/whitespace-only *feedback* -> return ``self`` unchanged.
        * Empty *content* -> return ``self`` unchanged (nothing to append to).
        """
        clean = feedback.strip()
        if clean.startswith("FEEDBACK"):
            clean = clean.replace("FEEDBACK from reasoning agent:", "").strip()
        if not clean or not self.content:
            return self
        return SIR(content=f"{self.content}\n\n--- REASONING FEEDBACK ---\n{clean}")

    def update(self, new_information: str) -> SIR:
        """Incremental update within translator inner loop.

        Replicates translator ``update_sir()`` semantics:
        * If content is empty -> set content to *new_information* directly.
        * Otherwise -> append with ``--- UPDATED SIR ---`` separator.
        """
        if not self.content:
            return SIR(content=new_information)
        return SIR(content=f"{self.content}\n\n--- UPDATED SIR ---\n{new_information}")

    def replace(self, new_content: str) -> SIR:
        """Full replacement of SIR content.

        Replicates ``_update_sir_history()`` line 80:
        ``self.current_sir = new_sir``.
        """
        return SIR(content=new_content)
