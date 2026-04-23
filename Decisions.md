# Decisions

## Source of Truth
This file records architectural decisions, tradeoffs, and reasoning behind non-obvious choices.
The coding agent must consult this file before making structural changes and must append a new
entry any time a meaningful architectural choice is made.

---

## Decision Log

### D-001 · Separate Round 2 inference file instead of modifying inference_gemini.py
**Date:** 2026-04-23
**Decision:** Created `inference_gemini_round2_schema_drift.py` as a copy-and-extend of
`inference_gemini.py` rather than modifying the original in place.
**Rationale:**
- Preserves a known-good baseline for regression comparison.
- Allows Round 2 changes to be isolated, reviewed, and reverted cleanly.
- Reduces risk of breaking existing hard/easy task evaluations during active development.
**Tradeoff:** Some code duplication between the two files. Acceptable for a hackathon timeline;
would warrant a shared utility module in a production setting.

---

### D-002 · Virtual action + normalization layer as the initial handle_drift approach
**Date:** 2026-04-23
**Decision:** First implemented `handle_drift` as a virtual action in the system prompt with a
normalization layer that mapped it to supported env actions, before later adding native support.
**Rationale:**
- The environment did not yet support `handle_drift` natively at the time of first implementation.
- A normalization layer allowed the agent to reason about drift handling immediately, without
  waiting for env-side changes.
- Decoupled agent reasoning from env capabilities during the iterative build.
**Tradeoff:** Added an extra translation layer that needed to be removed once native support
landed. Worth it for the unblocked parallel progress.

---

### D-003 · Native handle_drift action added to models.py and environment.py
**Date:** 2026-04-23
**Decision:** Promoted `handle_drift` from a virtual/normalized action to a first-class native
action registered in `src/models.py` and dispatched in `src/environment.py`.
**Rationale:**
- Native actions are cleaner, cheaper to reason about for the model, and avoid normalization bugs.
- Required for correct reward attribution — virtual actions can obscure whether the model
  actually selected the right action or was rerouted by the normalization layer.
- Enables proper RL credit assignment during GRPO training.
**Tradeoff:** Required coordinated changes across models.py, environment.py, and all three
inference files. Managed via py_compile validation after each change.

---

### D-004 · Fallback changed from run_pipeline to compare_schema
**Date:** 2026-04-23
**Decision:** Changed the default fallback action (when the agent is uncertain) from
`run_pipeline` to `compare_schema`.
**Rationale:**
- `run_pipeline` on a drifted schema produces misleading results and can trigger the
  −0.5 blind-fix penalty if downstream actions are based on stale column assumptions.
- `compare_schema` is a safe diagnostic action that surfaces drift signals without
  mutating state or incurring heavy penalties.
- Aligns with the reward structure: the env rewards correct diagnosis before action.
**Tradeoff:** Slightly more conservative agent behavior; may increase episode length.
Acceptable given the reward asymmetry (heavy penalty for blind fixes vs. small penalty for
extra diagnostic steps).

---

### D-005 · Run-indexed drift scheduler (_apply_scheduled_drift) triggered inside run_pipeline
**Date:** 2026-04-23
**Decision:** Implemented schema drift as a run-indexed schedule inside `_apply_scheduled_drift`,
called automatically during `run_pipeline` rather than as an external pre-step.
**Rationale:**
- Drift should be invisible to the agent at trigger time — the agent discovers it by observing
  pipeline failures, not by being told drift occurred. This is the realistic setting.
- Embedding it inside `run_pipeline` ensures the drift happens exactly when it would in
  production (mid-operation, not pre-announced).
- A seeded schedule makes episodes reproducible for reward variance analysis.
**Tradeoff:** Drift is harder to test in isolation. Mitigated by the smoke test suite that
verifies each scheduled event fires on the correct run index.

---

### D-006 · hard2 task added without modifying the existing hard task
**Date:** 2026-04-23
**Decision:** Added a new `hard2` task in `src/tasks.py` with its own drift schedule
(column rename → auth format rotation → rate limit tightening) while leaving the original
`hard` task completely unchanged.
**Rationale:**
- Preserves the existing hard task as a stable benchmark for comparing Round 1 vs Round 2
  model behavior.
- A separate task allows hard2-specific drift events to be tuned without risking regressions
  on the original evaluation suite.
- Keeps the task registry additive, which is consistent with the Instructions.md rule of
  preserving existing behavior unless explicitly required to change it.
**Tradeoff:** Two overlapping hard tasks increase maintenance surface. Acceptable for the
hackathon scope; would consolidate in a production codebase via parameterized task configs.

---

### D-007 · assertions.py updated to report missing-column failures explicitly
**Date:** 2026-04-23
**Decision:** Modified `src/assertions.py` to surface missing-column errors with explicit
failure messages rather than generic pipeline errors.
**Rationale:**
- Without explicit reporting, a column-rename drift event produces a cryptic KeyError that the
  agent cannot distinguish from an unrelated pipeline bug.
- Explicit messages give the model a learnable signal: "column X not found" → run
  `compare_schema` → detect rename → call `handle_drift(resolve_column_rename)`.
- This is essential for the training delta: the agent must have a clear observation to condition
  its corrective action on.
**Tradeoff:** Slightly more verbose assertion output. No functional downside.

---

### D-008 · pipeline_runner.py tolerates spend/total_spend in joins
**Date:** 2026-04-23
**Decision:** Updated `src/pipeline_runner.py` to accept either `spend` or `total_spend` as
valid column names in join operations, rather than hardcoding one.
**Rationale:**
- Allows the pipeline to run correctly both before and after the column-rename drift event,
  making it possible to validate pre-drift and post-drift behavior in the same test harness.
- Avoids having to maintain two separate pipeline configs for the same logical operation.
**Tradeoff:** Adds a small conditional in the join logic. Kept minimal and clearly commented.

---

### D-009 · OpenEnv stdout logging specification and terminal sanitization
**Date:** 2026-04-23
**Decision:** Implemented strict OpenEnv stdout formatting (`[START]`, `[STEP]`, `[END]`) and moved all other output to `sys.stderr`. Also removed all emojis from the codebase.
**Rationale:**
- The OpenEnv auto-grader strictly parses `sys.stdout`. Verbose logging and interactive outputs break the parser.
- Score clipping `min(max(score, 0.01), 0.99)` aligns with spec requirements.
- Emojis in python terminal output frequently cause `UnicodeEncodeError` on default Windows consoles. Replacing them with text markers like `[WARNING]` guarantees cross-platform stability.
**Tradeoff:** Loss of visual emojis in the terminal output, but traded for 100% stability without requiring users to tweak `PYTHONIOENCODING`.

---

## Template for New Entries

### D-XXX · Short title
**Date:** YYYY-MM-DD
**Decision:** What was decided.
**Rationale:** Why this was the right call given the constraints.
**Tradeoff:** What was given up or deferred.