# Security & Risk Acceptance Record

This document is the v1.0 honest-disclosure record for the SeeingEye LangGraph
migration. It consolidates three distinct risk categories that the user
explicitly accepted during planning. It is NOT a CVE advisory or a
vulnerability report — it is a migration audit artifact.

## 1. Credential History (Phase 1 skip, 2026-04-11)

Phase 1 "Security Cleanup" was **skipped** by user decision on 2026-04-11
(private repo, residual billing risk explicitly accepted). Historical private
keys were referenced across four hits in the old `src/multi-agent/` tree.
Literal values have been redacted from this public bundle:

- **OpenAI key** — `src/multi-agent/config/config.toml:5`
  (1 historical hit).
- **DashScope key** — 3 historical hits:
  - `src/multi-agent/config/config.toml:98`
  - `src/multi-agent/config/config.toml:203`
  - `src/multi-agent/benchmark_evaluation/ocrbench/eval_qwen.py:128`
  Literal values are not present in this public working tree.

**Deletion scope (Phase 7 DEL-01, 2026-04-17):**

- The two `config.toml` hits are deleted from `HEAD` — the old tree no longer
  exists in the working tree.
- The third DashScope hit (`eval_qwen.py:128`) has been replaced with
  `DASHSCOPE_API_KEY` environment-variable lookup in this public bundle.
- Historical private git history is not included in this public bundle.

**Additional live secret, NOT deleted by Phase 7:**

- **Azure OCR subscription key** — `src/seeingeye/tools/ocr.py` now reads
  `AZURE_OCR_SUBSCRIPTION_KEY` from the environment. No literal key is stored
  in this public working tree.

**If the repo ever goes public** or the risk disposition changes, re-open
Phase 1 via `/gsd:execute-phase 1` against the preserved plans in
`.planning/phases/01-security-cleanup/`. Rotation and `git filter-repo`
would both be in-scope at that point.

## 2. Empirical Validation Gap (Phase 6 deferral, 2026-04-17)

Phase 6 "Benchmark Adapter & Parity Validation" was **deferred** by user
decision on 2026-04-17. The ~24-run cluster benchmark sweep (3 runs of the
old code + 3 runs of the new code, each across 4 benchmarks) was pushed to
a post-migration follow-up to unblock Phase 7.

**Accepted risk:** The new `src/seeingeye/` tree is **NOT formally validated
against the paper's published accuracies:**

| Benchmark | Paper-reported | New-tree validated? |
|-----------|----------------|---------------------|
| MMMU val | 60.78% | NO |
| MMMU-Pro (standard) | 44.62% | NO |
| MMMU-Pro (vision) | 33.33% | NO |
| OCR-BenchV2 | 33.99% | NO |
| MIA-Bench | 84.10% | NO |

**What has been demonstrated:** 146 structural tests pass (Phase 2-5 scope,
post-Phase-7 drift-guard trim). These are unit + integration tests of the
new tree's internal contracts (state shapes, tool signatures, graph wiring,
import layering) — they do NOT measure accuracy on any benchmark.

**What has NOT been demonstrated:** That the new tree reproduces the
paper's Algorithm 1 behavior end-to-end on real benchmark inputs. The
project's original "Core Value" — "reproduces paper benchmark numbers" —
is **architecturally delivered (code-complete) but empirically unverified**.

See also: `.planning/ROADMAP.md` Phase 6 DEFERRED note for the full
deferral rationale.

## 3. Unverified Deviations From Old Code

At least one intentional deviation from `src/multi-agent/` semantics exists
in `src/seeingeye/` that Phase 6 parity validation would have surfaced:

- **OR-04-02-01: `force_answer_node` binds only `terminate_and_answer`**
  (`src/seeingeye/agents/reasoner/force_answer.py`). The old code
  (`src/multi-agent/app/agent/toolcall.py:252-285` + force-iteration
  prompt-swap) left all three decision tools bound
  (`terminate_and_answer`, `terminate_and_ask_translator`,
  `continue_reasoning`) and relied on prose in `FINAL_ITERATION_PROMPT`
  to disable the ask-translator tool. The new code binds only
  `terminate_and_answer` — structurally forcing the answer decision.

  This is a CORRECTNESS improvement under the paper's intended behavior
  ("return an answer on iter 3"), but it is a BEHAVIORAL change that
  Phase 6 would have validated. If the deferred parity sweep ever shows
  the new tree under-performs the paper, the first rollback target is:
  revert this function to bind all three tools and rely on prose-only
  force.

  Source: `.planning/phases/04-agents/04-RESEARCH.md` Finding #5,
  lines 246-258. Commit where this was introduced: see
  `.planning/phases/04-agents/04-02-SUMMARY.md`.

Any other deviations introduced during the Phase 2-5 migration are
documented in the per-phase `*-SUMMARY.md` files under `.planning/phases/`.

## 4. Recovery Path

If parity validation is ever re-opened:

1. **The old code is recoverable from git history.** The commit immediately
   preceding the Phase 7 DEL-01 deletion commit contains the full
   `src/multi-agent/` tree. Recover with:

   ```bash
   git log --all --pretty=oneline -- src/multi-agent/app/flow/iterative_refinement.py | head -5
   git checkout <pre-phase-7-commit> -- src/multi-agent/
   ```

2. **Re-open Phase 6** via `/gsd:plan-phase 6` — the FlowExecutor adapter
   (PAR-01) still needs to be built; the baseline sweep (PAR-02) needs
   cluster time; the harness cutover (PAR-04) is one-line-per-harness.

3. **If credential exposure risk changes** (repo goes public, key reuse
   detected), re-open Phase 1 via `/gsd:execute-phase 1`. Rotation at
   provider consoles is a user-only manual step; `git filter-repo` would
   rewrite history if the public-repo branch of D-11 is taken.

---

*Last updated: 2026-04-17 — Phase 7 DEL-03.*
