"""Paper-frozen prompts for the SeeingEye flow.

All prompt files in this package are verbatim copies of the corresponding
files in src/multi-agent/app/prompt/. Each carries a `# DO NOT EDIT --
paper-frozen` header. Drift detection is delegated to the Phase 6 parity
gate (TPS-06 SHA-256 hash check was cut as over-engineered).
"""

from src.seeingeye.prompts import translator, reasoner, vqa_mmmu, force_answer

__all__ = ["translator", "reasoner", "vqa_mmmu", "force_answer"]
