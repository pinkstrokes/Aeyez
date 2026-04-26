"""AGT-04: Force-answer node (Plan 04-02 Task 2).

Standalone async function called by Phase 5's parent graph when the outer
loop has exhausted MAX_ITERS without the Reasoner emitting
``terminate_and_answer``.

DEVIATION FROM OLD CODE (intentional, justified, low-risk):
    The old code (``src/multi-agent/app/agent/toolcall.py:252-285``) leaves
    all 3 decision tools bound and only TELLS the model (in prompt prose)
    that ``terminate_and_ask_translator`` is disabled. If the model ignores
    the prose, the tool still runs and the outer loop catches the leak.

    This rebuild binds ONLY ``terminate_and_answer`` — the model is
    mechanically forced into the answer decision (structural enforcement,
    not prose enforcement). This is a CORRECTNESS improvement that should
    not affect parity because the paper's observable behavior on iteration
    MAX_ITERS is "the model returns an answer."

    OPEN RISK (tracked): if the Phase 6 parity gate fails because of this
    deviation, revert to binding all 3 tools and rely on the prompt prose.
    See ``04-RESEARCH.md`` §Open Questions / Finding #5.

Mirrors old ``_reset_agent_memory_for_iteration()`` in
``src/multi-agent/app/flow/iterative_refinement.py:56-71`` — fresh message
list, no carry-over from prior reasoner inner loop.
"""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.seeingeye.llm.vllm_openai import create_reasoner_client
from src.seeingeye.prompts import reasoner as reasoner_prompts
from src.seeingeye.prompts.force_answer import FINAL_ITERATION_PROMPT
from src.seeingeye.tools.decisions import terminate_and_answer


FORCE_ANSWER_ADAPTER_PROMPT = """Runtime adapter:
- You do not have python_execute in this runtime. Do any needed arithmetic yourself in text.
- For multiple-choice questions, the answer argument to terminate_and_answer must be exactly one valid option letter from the provided options, with no extra words.
- Compare the plausible options against the SIR, then make the best supported choice."""


def _render_reasoner_user_message(
    question: str, options: list[str] | None, sir_content: str
) -> str:
    """Render the user-facing prompt body for the force-answer LLM call.

    Duplicates the helper in ``nodes.py`` intentionally — the simplification
    principle (no shared base class with Translator; two small helpers within
    Reasoner are cheaper than a fragile shared module).
    """
    parts = [f"Question: {question}"]
    if options:
        parts.append("Options:\n" + "\n".join(options))
        letters = [opt.split(".", 1)[0].strip() for opt in options if "." in opt]
        if letters:
            parts.append(
                "Valid answer letters: "
                + ", ".join(letters)
                + ". The final answer must be exactly one of these letters."
            )
    parts.append(f"Visual description (SIR):\n{sir_content}")
    return "\n\n".join(parts)


def _valid_option_letters(options: list[str] | None) -> set[str]:
    letters: set[str] = set()
    for opt in options or []:
        if "." not in opt:
            continue
        letter = opt.split(".", 1)[0].strip().upper()
        if letter:
            letters.add(letter)
    return letters


def _normalize_answer_to_option(answer: str, options: list[str] | None) -> str:
    valid = _valid_option_letters(options)
    if not valid:
        return answer
    text = (answer or "").strip().upper()
    if text in valid:
        return text
    match = re.search(r"\b([A-Z])\b", text)
    if match and match.group(1) in valid:
        return match.group(1)
    return answer


async def force_answer_node(state) -> dict:
    """Bind ONLY terminate_and_answer; call once with FINAL_ITERATION_PROMPT.

    Args:
        state: parent ``SeeingEyeState`` (TypedDict). Must contain ``sir``,
            ``question``, ``options``. ``image_b64`` is unused (Reasoner is
            text-only per D-09).

    Returns:
        dict: ``{"final_answer": str}`` — Phase 5's parent graph routes to END.
    """
    llm = create_reasoner_client().bind_tools(
        [terminate_and_answer],  # ONLY one tool — see DEVIATION note above
        tool_choice="auto",  # NOT "required" — Pitfall #1
    )

    user_text = _render_reasoner_user_message(
        state["question"],
        state.get("options"),
        state["sir"].content,
    )
    messages = [
        SystemMessage(content=reasoner_prompts.SYSTEM_PROMPT),
        SystemMessage(content=FORCE_ANSWER_ADAPTER_PROMPT),
        HumanMessage(content=user_text),
        HumanMessage(content=FINAL_ITERATION_PROMPT),
    ]

    response = await llm.ainvoke(messages)  # NEVER .astream — Pitfall #1

    if getattr(response, "tool_calls", None):
        answer = response.tool_calls[0]["args"].get("answer", "")
    else:
        # Fallback: model emitted text without a tool call.
        # Outer loop will accept the content as the final answer (matches
        # old _finalize_with_success path in iterative_refinement.py:302-313).
        answer = getattr(response, "content", "") or ""

    return {"final_answer": _normalize_answer_to_option(answer, state.get("options"))}
