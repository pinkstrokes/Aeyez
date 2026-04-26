# DO NOT EDIT — paper-frozen
# Source: src/multi-agent/app/prompt/text_only_reasoning.py (FINAL_ITERATION_PROMPT)
"""Force-answer prompt for the AGT-04 force-answer node.

Extracted verbatim from text_only_reasoning.FINAL_ITERATION_PROMPT so that
Phase 4 has a clean import surface: `from src.seeingeye.prompts.force_answer
import FINAL_ITERATION_PROMPT`.
"""

FINAL_ITERATION_PROMPT = """🚨 **FINAL ITERATION** - You MUST provide a definitive answer now. The terminate_and_ask_translator tool is DISABLED.

ALWAYS provide your reasoning and thoughts BEFORE taking any action.

Consider these final evaluation points:
- Does the problem require calculations, data analysis, or computational verification?
- Does the visual description provide specific, distinguishing details?
- Can you clearly differentiate between all options based on the description?
- You MUST choose an answer - either with high confidence or your best educated guess

🔧 **COMPUTATION NEEDED** - USE python_execute FIRST:
   - When math/data processing clarifies the answer.
   - Need to verify calculations or process numerical information
   - **ALWAYS** include print() statements to show your work and results

🟢 **MUST USE terminate_and_answer** (this is your ONLY option):
   - **HIGH CONFIDENCE (>90%)**: You can clearly rule out incorrect options and are confident in your answer
   - **BEST GUESS (<90%)**: If you are not confident, you MUST still guess the best match option based on your current analysis
   - **MANDATORY**: Your answer must match one of the multiple choice options (A, B, C, D) if applicable
   - **IMPORTANT**: If your calculated answer doesn't match any option, use python_execute again to recalculate with different approach/units/interpretation
   - Explain your reasoning and confidence level in your answer

Keep responses under 1024 tokens - be concise and focus on key reasoning points.
"""
