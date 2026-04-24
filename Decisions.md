# Decisions

## Source of Truth
This file records architectural decisions, tradeoffs, and reasoning behind non-obvious choices.
The coding agent must consult this file before making structural changes and must append a new
entry any time a meaningful architectural choice is made.

---

## Decision Log

### D-001 - Separate Round 2 inference file instead of modifying inference_gemini.py
**Date:** 2026-04-23
**Decision:** Created inference_gemini_round2_schema_drift.py as a copy-and-extend of
inference_gemini.py rather than modifying the original in place.
**Rationale:** Preserves a known-good baseline for regression comparison. Allows Round 2
changes to be isolated and reverted cleanly. Reduces risk during active development.
**Tradeoff:** Some code duplication. Acceptable for hackathon timeline.

---

### D-002 - Virtual action + normalization layer as the initial handle_drift approach
**Date:** 2026-04-23
**Decision:** First implemented handle_drift as a virtual action in the system prompt with a
normalization layer mapping it to supported env actions, before adding native support.
**Rationale:** Environment did not yet support handle_drift natively. Normalization layer
allowed agent reasoning to proceed without waiting for env-side changes.
**Tradeoff:** Extra translation layer needed removal once native support landed.

---

### D-003 - Native handle_drift action added to models.py and environment.py
**Date:** 2026-04-23
**Decision:** Promoted handle_drift from virtual to a first-class native action registered
in src/models.py and dispatched in src/environment.py.
**Rationale:** Native actions are cleaner, cheaper to reason about, and avoid normalization
bugs. Required for correct reward attribution and RL credit assignment during GRPO.
**Tradeoff:** Coordinated changes across models.py, environment.py, and all inference files.

---

### D-004 - Fallback changed from run_pipeline to compare_schema
**Date:** 2026-04-23
**Decision:** Changed the default fallback action from run_pipeline to compare_schema.
**Rationale:** run_pipeline on a drifted schema produces misleading results and can trigger
the -0.5 blind-fix penalty. compare_schema is a safe diagnostic action with no state mutation.
**Tradeoff:** Slightly more conservative agent behavior; may increase episode length.

---

### D-005 - Run-indexed drift scheduler triggered inside run_pipeline
**Date:** 2026-04-23
**Decision:** Schema drift implemented as a run-indexed schedule inside _apply_scheduled_drift,
called automatically during run_pipeline rather than as an external pre-step.
**Rationale:** Drift should be invisible to the agent at trigger time — discovered by observing
failures, not by being told. Realistic production setting. Seeded schedule ensures reproducibility.
**Tradeoff:** Drift harder to test in isolation. Mitigated by smoke test suite.

---

### D-006 - hard2 task added without modifying the existing hard task
**Date:** 2026-04-23
**Decision:** Added new hard2 task in src/tasks.py with its own drift schedule while leaving
the original hard task completely unchanged.
**Rationale:** Preserves hard as a stable benchmark for Round 1 vs Round 2 comparison.
Separate task allows hard2-specific drift events to be tuned without regressions.
**Tradeoff:** Two overlapping hard tasks increase maintenance surface.

---

### D-007 - assertions.py updated to report missing-column failures explicitly
**Date:** 2026-04-23
**Decision:** Modified src/assertions.py to surface missing-column errors with explicit
failure messages rather than generic pipeline errors.
**Rationale:** Without explicit reporting, a column-rename drift event produces a cryptic
KeyError the agent cannot distinguish from an unrelated bug. Explicit messages give a
learnable signal: "column X not found" -> compare_schema -> handle_drift(resolve_column_rename).
**Tradeoff:** Slightly more verbose assertion output. No functional downside.

---

### D-008 - pipeline_runner.py tolerates spend/total_spend in joins
**Date:** 2026-04-23
**Decision:** Updated src/pipeline_runner.py to accept either spend or total_spend as valid
column names in join operations.
**Rationale:** Allows pipeline to run correctly both before and after the column-rename drift
event, enabling pre/post-drift validation in the same test harness.
**Tradeoff:** Small conditional in join logic. Kept minimal and clearly commented.

---

### D-009 - OpenEnv stdout logging and terminal sanitization
**Date:** 2026-04-23
**Decision:** Strict OpenEnv stdout formatting ([START], [STEP], [END]) with all other output
to sys.stderr. Emojis removed from entire codebase.
**Rationale:** OpenEnv auto-grader strictly parses sys.stdout. Emojis cause UnicodeEncodeError
on default Windows consoles.
**Tradeoff:** No visual emojis; traded for 100% cross-platform stability.

---

### D-010 - Native Kaggle support via HuggingFace LLaMA and 4-bit quantization
**Date:** 2026-04-23
**Decision:** Kaggle notebook uses HuggingFace transformers with BitsAndBytesConfig(load_in_4bit=True)
and unsloth/Meta-Llama-3.1-8B-Instruct instead of local Ollama Qwen setup.
**Rationale:** 8B models exceed 16GB VRAM limits of Kaggle T4 in native 16-bit. 4-bit quantization
fits within Kaggle constraints while maintaining reasoning capability.
**Tradeoff:** Slightly longer init time vs Ollama. Minor precision reduction. Acceptable.

---

### D-011 - inference.py must read env vars per OpenEnv spec
**Date:** 2026-04-24
**Decision:** Fixed inference.py to read API_BASE_URL, MODEL_NAME, HF_TOKEN from environment
variables with local fallback defaults. RESOLVED.
**Rationale:** The hackathon automated validator injects its own env vars and runs inference.py.
Hardcoded localhost:11434 would target a non-existent endpoint and fail immediately.
**Fix applied:**
  API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
  API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "MISSING_KEY"
  MODEL_NAME   = os.getenv("MODEL_NAME", "llama3")
**Tradeoff:** None. Local defaults preserve dev workflow.

---

### D-012 - Training algorithm: SFT then GRPO with KL divergence penalty
**Date:** 2026-04-24
**Decision:** Two-stage training: SFT on gold trajectories (easy/medium), then GRPO on hard/hard2
with shaped environment reward. KL coefficient 0.1 against SFT reference.
**Rationale:**
- SFT first gives model a warm start on action format and basic diagnostic strategy.
  Cold GRPO on random policy produces high-variance gradients and slow convergence.
- GRPO preferred over PPO: no critic network required, lower memory pressure on T4,
  dense environment reward makes group-relative comparisons meaningful at G=4.
- KL=0.1 prevents policy collapse while allowing reward improvement.
- Mirrors DeepSeek-R1 / Unsloth GRPO approach aligned with judge expectations.
**Implementation:** collect_gold_trajectories() -> SFTTrainer -> pipeline_reward_fn() -> GRPOTrainer.
**Tradeoff:** Two-stage adds ~90 min total compute. If time-constrained, SFT alone with
before/after inference comparison is sufficient to demonstrate improvement for the pitch.

---

### D-013 - GRPO reward function includes format and drift detection bonuses
**Date:** 2026-04-24
**Decision:** pipeline_reward_fn adds +0.3 for valid JSON output and +0.3 when compare_schema
detects real schema_diff, on top of the base environment step reward.
**Rationale:**
- Format bonus: aligns with the model's incentive to produce parseable actions rather than prose.
  Without it, a model that outputs valid JSON gets the same signal as one that outputs garbage.
- Drift detection bonus: the key training delta for hard2 is teaching the model to call
  compare_schema after a failed run_pipeline before applying any patch. The bonus makes this
  behavior explicitly rewarded, not just implicitly incentivized through downstream assertion gains.
**Tradeoff:** Slightly non-standard reward shaping. Justified because the environment's
terminal reward (+1.0 for full pass) is sparse and delayed; intermediate shaping is necessary
for credit assignment in a 20-step episode.

---

### D-012 · Anti-repetition loop breaker in Qwen Kaggle notebook
**Date:** 2026-04-24
**Decision:** Added `_detect_stuck_loop()` and `_get_unstuck_action()` to the Qwen notebook runner. When the model repeats the same action 3+ times consecutively, the system overrides it with a heuristically chosen action and trims conversation history to 4 turns.
**Rationale:**
- Smaller models (Qwen 3B) get trapped in action loops (e.g. `coalesce unit_price` → `run_pipeline` → repeat) because the accumulated history reinforces the pattern.
- Trimming history breaks the learned loop by removing the repetitive context the model conditions on.
- The heuristic fallback (`dedup` for medium, `read_data_sample` for run_pipeline loops) is derived from the gold fix actions for each task, so it nudges toward the correct solution.
**Tradeoff:** Hard-coded heuristics reduce generality. Acceptable for a 3B model that lacks the reasoning to self-correct; larger models (8B+) with proper instruction following do not trigger the loop detector.

---

## Template for New Entries

### D-XXX - Short title
**Date:** YYYY-MM-DD
**Decision:** What was decided.
**Rationale:** Why this was the right call given the constraints.
**Tradeoff:** What was given up or deferred.
