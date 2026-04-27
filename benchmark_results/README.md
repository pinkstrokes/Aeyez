# Aeyez Benchmark Results

This folder keeps measured results from local Aeyez / Spaz evaluation runs.
These are project validation artifacts, not official leaderboard submissions.

## MMMU Custom Hardset

- Result file: [`mmmutest.jsonl`](./mmmutest.jsonl)
- Total questions: 250
- Correct: 220
- Accuracy: 88.0%
- Evaluation style: open-answer adjusted, using the `correct` field recorded
  in the JSONL.
- Run makeup:
  - `original_run`: 201 questions
  - `relevance_replacement_v1`: 26 questions
  - `vstar_unfinished_rerun`: 18 questions
  - `mechanics_enhanced_rerun_improvement`: 5 questions

The set was curated around visual reasoning that matters for Aeyez: spatial
relationships, route and obstruction reasoning, engineering/mechanics-style
visual verification, local crop/zoom comparison, and safety-relevant scene
understanding.
