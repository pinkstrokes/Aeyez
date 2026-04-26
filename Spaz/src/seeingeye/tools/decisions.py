"""Reasoner decision actions for the SeeingEye flow.

TPS-04: defined as langchain-core ``@tool`` functions so the Reasoner
(Qwen3-8B on vLLM port 8001) can ``bind_tools([...])`` them and emit
OpenAI-compatible tool calls.

The Translator side uses parsed-VCoT text sentinels in
``agents/translator/policy.py`` (NOT bound via vLLM tool calls on
Qwen2.5-VL — see Pitfall #1 / AGT-01).

Descriptions for the two ported decisions are COPIED VERBATIM from
``src/multi-agent/app/tool/terminate_and_answer.py`` and
``src/multi-agent/app/tool/terminate_and_ask_translator.py``.  Do not
paraphrase — the Reasoner has been trained against those exact strings.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool


@tool
def terminate_and_answer(
    answer: str,
    confidence: Literal["high", "medium", "low"],
    reasoning: str,
) -> str:
    """Terminate the reasoning process and provide a final answer when you have sufficient information from the SIR to confidently answer the question.

    Use this tool when:
    - The SIR contains all necessary visual details to answer the question
    - You can identify the correct answer from the available options
    - No additional information or refinement is needed from the translator agent
    - Your answer matches one of the multiple choice options (if applicable)

    IMPORTANT: For multiple choice questions, ensure your answer corresponds to one of the given options (A, B, C, D).

    This signals that the iterative feedback loop should end with your final answer.

    Args:
        answer: Your final answer to the question. Please include short answer only. For multiple choice, only include option.
        confidence: Your confidence level in this answer.
        reasoning: Brief explanation of how the SIR information led to this answer.
    """
    return (
        f"FINAL ANSWER: {answer}\n\n"
        f"Confidence: {confidence}\n\n"
        f"Reasoning: {reasoning}\n\n"
        f"The reasoning process has been completed successfully."
    )


@tool
def terminate_and_report_safety(
    hazards: str,
    safest_next_action: str,
    safest_route: str,
    route_risks: str,
    route_now_vs_after_clear: str,
    safest_solution: str,
    confidence: Literal["high", "medium", "low"],
    reasoning: str,
    current_scene: str = "",
    dynamic_clear_scene: str = "",
    route_candidates: str = "",
    candidate_risk_scores: str = "",
    no_go_zones: str = "",
    verification_needed: str = "",
    safest_now: str = "",
    safest_after_clear: str = "",
    fallback: str = "",
) -> str:
    """Terminate the reasoning process with a safety-focused assessment.

    Use this tool when:
    - The task is safety-oriented and you can identify meaningful nearby hazards
    - You can recommend the safest next action supported by the SIR
    - You can predict the safest visible or inferable route through the environment
    - You can describe the safest practical solution or operating approach
    - Additional translator refinement is not required for a useful safety recommendation

    Args:
        hazards: Concise description of the main safety hazards or precursor conditions.
        safest_next_action: The immediate next action that best reduces risk.
        safest_route: The safest route or movement path to take, including key landmarks or areas to avoid.
        route_risks: Route-specific risks, blocked paths, blind spots, unstable areas, moving equipment, or uncertainty.
        route_now_vs_after_clear: How the route differs now versus after temporary workers, carried materials, vehicles, or other dynamic obstructions clear.
        safest_solution: The safest complete way to proceed or resolve the situation.
        confidence: Your confidence level in this recommendation.
        reasoning: Brief explanation connecting the visible evidence to the recommendation.
        current_scene: Structured summary of the scene as it is now, including dynamic and static blockers.
        dynamic_clear_scene: Structured before/after summary of what may change after movable blockers clear.
        route_candidates: Candidate routes retained for comparison.
        candidate_risk_scores: Qualitative risk scoring for candidate routes.
        no_go_zones: Areas that should not be entered under current conditions.
        verification_needed: Conditions that must be checked before using uncertain paths.
        safest_now: Safest immediate action in the current scene.
        safest_after_clear: Best route/action after dynamic obstructions clear and checks pass.
        fallback: Backup plan if the after-clear route is not verified safe.
    """
    return (
        "SAFETY REPORT\n\n"
        f"Current scene: {current_scene}\n\n"
        f"Dynamic-clear scene: {dynamic_clear_scene}\n\n"
        f"Route candidates: {route_candidates}\n\n"
        f"Candidate risk scores: {candidate_risk_scores}\n\n"
        f"No-go zones: {no_go_zones}\n\n"
        f"Verification needed: {verification_needed}\n\n"
        f"Safest now: {safest_now}\n\n"
        f"Safest after clear: {safest_after_clear}\n\n"
        f"Fallback: {fallback}\n\n"
        f"Hazards: {hazards}\n\n"
        f"Safest next action: {safest_next_action}\n\n"
        f"Safest route: {safest_route}\n\n"
        f"Route risks: {route_risks}\n\n"
        f"Route now vs after clear: {route_now_vs_after_clear}\n\n"
        f"Safest solution: {safest_solution}\n\n"
        f"Confidence: {confidence}\n\n"
        f"Reasoning: {reasoning}\n\n"
        "The safety assessment has been completed successfully."
    )


@tool
def terminate_and_ask_translator(feedback: str) -> str:
    """Terminate current reasoning step and request more specific visual observations from the translator.

    Use this tool when:
    - The current SIR (visual description) is insufficient for answering the question
    - You need more specific details about certain parts of the image
    - Important visual elements seem to be missing from the description
    - You need clarification about spatial relationships, text content, or visual elements
    - The translator's description lacks crucial information needed for reasoning

    This signals that you need additional visual analysis before you can provide a final answer.

    Args:
        feedback: Specific feedback about what additional visual information you need from the translator. Be precise about what's missing or unclear in the current description.
    """
    return f"feedback: {feedback}"


@tool
def continue_reasoning(thought: str) -> str:
    """Continue the reasoning loop without terminating and without asking the translator for more visual information.

    Use this tool when:
    - You need to do additional internal reasoning or computation before answering
    - The visual information is sufficient but the logical chain is not yet complete
    - You want to verify a calculation or cross-check facts before terminating

    Args:
        thought: A short description of what additional reasoning step you intend to perform.
    """
    return f"continuing: {thought}"
