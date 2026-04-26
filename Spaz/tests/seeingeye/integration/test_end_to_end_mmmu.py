"""GRF-05 end-to-end sanity: run_question on ONE hand-picked MMMU val question.

Pass criteria (REQUIREMENTS.md GRF-05 simplified 2026-04-13):

- ``answer`` contains the known ground truth letter.
- ``total_tokens > 0`` (proof that LLMs were actually called).
- ``outer_iters_used`` in ``{1, 2, 3}`` (within ``MAX_ITERS`` range).

**NOT an accuracy test** — that is Phase 6 (PAR-02..07). This test only
proves the wiring works end-to-end.

vLLM prerequisite: both endpoints (translator :8000, reasoner :8001)
must be up. If either is unreachable, :func:`pytest.skip` — do NOT fail
— per 05-CONTEXT.md §specifics. Phase 6 PAR-02 re-runs the full sweep
and will catch any wiring regression there.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

from src.seeingeye.config.settings import Settings
from src.seeingeye.runtime import SeeingEyeResult, run_question


# ---------------------------------------------------------------------------
# Hand-picked MMMU val sample (audit trail per plan §acceptance criteria).
#
# - Sample id:       validation_Art_1 (MMMU_DEV_VAL index 1560)
# - Subfield:        Art / Drawing and Painting
# - Difficulty:      Easy (multiple-choice)
# - Image:           extracted from MMMU_DEV_VAL.tsv base64 to
#                    tests/seeingeye/integration/fixtures/validation_Art_1.jpg
# - Ground truth:    C (Hermione)
# - Rationale:       Easy-difficulty multi-choice question from the MMMU val
#                    split — small enough that SeeingEye's outer loop should
#                    converge within 1..3 iterations; the correct answer is a
#                    single letter so the lenient containment check in the
#                    assertion is robust to minor formatting variance.
# ---------------------------------------------------------------------------
MMMU_IMAGE_PATH = Path("tests/seeingeye/integration/fixtures/validation_Art_1.jpg")
MMMU_QUESTION = (
    "This Roman portrait mummy <image 1> from the 1st century AD was discovered "
    "in Cairo, Egypt, and now belongs in the collection of Girton College, "
    "University of Cambridge. Which well-known fictional character does this "
    "woman share her name with?"
)
MMMU_OPTIONS: list[str] | None = [
    "A. Aurelia",
    "B. Matilda",
    "C. Hermione",
    "D. Juno",
]
MMMU_GROUND_TRUTH = "C"


def _is_endpoint_reachable(base_url: str, timeout_s: float = 2.0) -> bool:
    """TCP-connect check on the vLLM endpoint host:port.

    NOT a full HTTP handshake — just a connect check. If the port is open
    we assume vLLM is up; the real HTTP call inside :func:`run_question`
    surfaces any deeper issues as test failures (not skips).
    """
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except (OSError, socket.timeout):
        return False


@pytest.mark.integration
def test_run_question_end_to_end_mmmu_sanity() -> None:
    """GRF-05: One MMMU val question through the full pipeline against live vLLM.

    SKIPs when either vLLM endpoint is unreachable (05-CONTEXT.md §specifics).
    """
    settings = Settings()

    # Prerequisite checks — skip if vLLM is down (not a test failure).
    if not _is_endpoint_reachable(settings.translator_base_url):
        pytest.skip(
            f"Translator vLLM not reachable at {settings.translator_base_url} — "
            "GRF-05 requires live vLLM. Start the endpoint or run Phase 5 "
            "locally with vLLM up. Phase 6 PAR-02 will re-run this as part "
            "of parity and catch any wiring regression there."
        )
    if not _is_endpoint_reachable(settings.reasoner_base_url):
        pytest.skip(
            f"Reasoner vLLM not reachable at {settings.reasoner_base_url} — "
            "GRF-05 requires live vLLM. Phase 6 PAR-02 will cover the gap."
        )

    if not MMMU_IMAGE_PATH.exists():
        pytest.skip(
            f"MMMU image fixture not found at {MMMU_IMAGE_PATH}. Regenerate "
            "via the extraction snippet in 05-01-SUMMARY.md."
        )

    result: SeeingEyeResult = asyncio.run(
        run_question(
            question=MMMU_QUESTION,
            image_path=MMMU_IMAGE_PATH,
            options=MMMU_OPTIONS,
        )
    )

    # Pass criteria (GRF-05 simplified).
    assert isinstance(result, SeeingEyeResult)
    assert result.total_tokens > 0, "total_tokens must be >0 — LLMs were called"
    assert 1 <= result.outer_iters_used <= 3, (
        f"outer_iters_used={result.outer_iters_used} outside paper range 1..3"
    )

    # Answer match — lenient case-insensitive containment. Full accuracy
    # judging is Phase 6's job with the proper harness metric logic.
    assert MMMU_GROUND_TRUTH.lower() in result.answer.lower(), (
        f"Answer '{result.answer}' does not contain ground truth "
        f"'{MMMU_GROUND_TRUTH}'. Final SIR:\n{result.sir.content}"
    )
