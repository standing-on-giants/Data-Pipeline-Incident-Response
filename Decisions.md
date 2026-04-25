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

### D-012 · Prompt-based loop detection instead of harness-level action override
**Date:** 2026-04-24
**Decision:** Replaced the harness-level `_detect_stuck_loop()` + `_get_unstuck_action()` approach (which silently overrode the model's chosen action) with a prompt-based approach: `_detect_action_loop()` parses `obs.actions_taken` (format: `[N] action_type({params})`) for 3+ identical actions or 2-step oscillation patterns. When detected, `_build_loop_hint()` injects a `[CRITICAL LOOP DETECTED]` section into the user prompt with assertion-specific fix suggestions. The runner trims history to 2 turns to break the repetitive context, but the model still decides its own next action.
**Rationale:**
- The original approach silently overrode the model's action, which masks the real problem (the model doesn't understand what to do) and prevents learning during GRPO training. If the harness takes the right action, the model gets reward for an action it never chose — poisoning the credit assignment.
- Prompt-based hints let the model self-correct: it sees "LOOP DETECTED: you keep repeating coalesce on unit_price... uniqueness failure → the ONLY fix is dedup" and can choose dedup on its own.
- History trimming (to 2 turns) is the critical complement — without it, the long repetitive context overwhelms the hint, and the small model falls back into the pattern.
- This approach also works correctly with GRPO training: the model learns to respond to loop hints, not depend on external rescue.
**Tradeoff:** If the model ignores the hint (possible with very weak models), the loop continues until max_steps. Acceptable because: (a) the hint is very explicit and assertion-specific, (b) history trimming breaks the pattern most of the time, (c) this preserves correct RL credit assignment.

---

### D-013 · Three-variant GRPO training architecture
**Date:** 2026-04-24
**Decision:** Implemented GRPO training as 3 separate scripts sharing the same reward function: (1) `train_grpo.py` general CLI, (2) `training_grpo_qwen.ipynb` for Qwen2.5-1.5B on Kaggle, (3) patched `training_grpo.ipynb` for LLaMA 8B with optional Gemini API trajectories.
**Rationale:**
- Different models have different VRAM profiles. Qwen 1.5B uses G=8 and r=32 (more samples + higher rank because the model is small). LLaMA 8B uses G=4 and r=16.
- Chose Qwen2.5-1.5B (text-only) over Qwen2.5-VL-3B for training because the vision encoder adds ~7GB VRAM overhead with zero benefit for this text-only task, and 1.5B is faster to fine-tune on a T4.
- The CLI script enables non-Kaggle training (local GPUs, cloud instances) and supports any HF model via `--model`.
- Reward function is identical across all variants: env_step_reward + format_bonus + drift_bonus + loop_penalty. This ensures trained models are comparable.
**Tradeoff:** Three separate files to maintain. Mitigated by using the same reward function and gold trajectory definitions.

---

### D-014 · Downgrade inference model to Qwen2.5-1.5B-Instruct
**Date:** 2026-04-24
**Decision:** Replaced `Qwen/Qwen2.5-VL-3B-Instruct` (Vision-Language) with `Qwen/Qwen2.5-1.5B-Instruct` (text-only) as the default lightweight inference model for Kaggle T4 notebooks.
**Rationale:**
- The task is entirely text-based. The VL model's vision encoder (~0.8 GB) occupies VRAM with zero benefit.
- 1.5B vs 3B: half the parameters, ~3x faster generation, ~0.8 GB vs ~2.5 GB VRAM at 4-bit NF4.
- Same Qwen family, same tokenizer behavior, no compatibility issues with existing prompt/parser code.
- Leaves 13+ GB VRAM headroom on T4 (vs 9+ GB for 3B-VL), nearly eliminating OOM risk on long episodes.
**Tradeoff:** Smaller model capacity may produce slightly weaker reasoning. Acceptable for inference evaluation; GRPO fine-tuning on this base should recover quality.

---

### D-015 · Fix 3 silent environment logic bugs instead of defensive notebook wrappers
**Date:** 2026-04-24
**Decision:** Fixed the root-cause bugs in `src/environment.py` directly (`mark_acceptable` override, `add_data_filter` operator validation, `read_data_sample` column guard) rather than adding defensive error-catching in notebooks/inference scripts.
**Rationale:**
- Silent failures break RL credit assignment: if the agent gets `0.0` for a no-op action, GRPO cannot learn which actions are effective.
- Fixing at the source makes all inference scripts, training scripts, and notebooks correct simultaneously without individual patching.
- The `mark_acceptable` bug was especially critical: it made it literally impossible for the agent to finish an episode via acceptance (episode always timed out). Correcting the reward from `-1.0` to `+0.1` makes the action semantically correct for the first time.
**Tradeoff:** Changing reward values is a behavioral change that could affect GRPO credit assignment for trajectories trained on the old env. Acceptable because the old values were incorrect and would have produced wrong gradient signals anyway.

---

### D-016 · Aggressive context management for Qwen 1.5B inference
**Date:** 2026-04-24
**Decision:** Updated the Kaggle 1.5B notebook runner to severely limit context memory: capping history to 6 turns, truncating user prompts at 3000 chars, and auto-wiping history entirely if 3 consecutive OOMs/generation errors occur.
**Rationale:**
- The 1.5B model's context window can handle the prompt initially, but as the episode drags on (e.g., 90 steps), accumulating history causes silent Out of Memory errors or context degradation.
- RL requires episodes to finish successfully or timeout gracefully, not crash on hardware constraints mid-way.
- Wiping the history when stuck acts as a "hard reset" that allows the agent to look at the current board state fresh without dying, ensuring we get a valid RL rollout.
**Tradeoff:** The agent loses memory of past actions when history is truncated or wiped, making it more prone to repeating actions. This is acceptable because the current state (failing assertions) usually holds enough signal to decide the next action, and completing the episode is better than an unhandled exception.

---

### D-017 · Reward shaping and smart fallback to break repetitive action loops
**Date:** 2026-04-24
**Decision:** Updated `src/environment.py` with explicit penalties for zero-progress pipeline runs (-0.15) and duplicate patches (-0.3). Also updated `create_qwen_1_5b_notebook.py` to use `read_data_sample` as the JSON-parsing fallback instead of `compare_schema`, and to reset `consecutive_errors` during loop-breaks.
**Rationale:**
- The 1.5B model frequently entered loops of re-running `run_pipeline` with no changes, or repeating identical patches (`coalesce` on `ctr`) endlessly. The environment gave `0.0` reward for these, offering no penalty signal.
- The default JSON-parsing fallback (`compare_schema`) exacerbated loops because it doesn't mutate state and just repeats indefinitely. `read_data_sample` is more useful for context when parsing fails.
- By providing explicit negative rewards for repeating actions, we enable GRPO training to penalize these cyclic paths, and by fixing the loop-breaking heuristics in the notebook runner, we force the inference rollout to escape local minima.
**Tradeoff:** Increased state tracking inside the environment (`_applied_patches_set`). Acceptable since it's cheap to track tuples of strings in memory.

---

### D-018 · Qwen 1.5B Prompt Tuning: Action History & Repetition Warnings
**Date:** 2026-04-25
**Decision:** Updated `create_qwen_1_5b_notebook.py` to feed the last 20 actions (instead of 5) into the `obs.actions_taken` section of the prompt. Additionally, added a nuanced rule to the `SYSTEM_PROMPT` explaining that while repeating actions is allowed, doing so without applying a new fix wastes step budget and incurs negative penalties.
**Rationale:**
- The 1.5B model struggled to escape loops because it could only "see" 5 steps into the past, losing track of its earlier failed attempts.
- An overly strict "NEVER repeat" rule might prevent the agent from legitimately re-running the pipeline after applying a fix. By providing a longer trajectory window (20 actions) and explicitly explaining that repeating actions is allowed *only if it makes sense* (but otherwise wastes budget), we give the model enough context and incentive to break out of its own repetitive cycles without breaking its ability to verify fixes.
**Tradeoff:** A slight increase in token usage per step due to the longer action history, but acceptable given the strict truncation applied elsewhere in the notebook runner.

---

## Template for New Entries

### D-XXX - Short title
**Date:** YYYY-MM-DD
**Decision:** What was decided.
**Rationale:** Why this was the right call given the constraints.
**Tradeoff:** What was given up or deferred.

### D-014 - Standalone Qwen Comparison Script via Transformers
**Date:** 2026-04-24
**Decision:** Developed a Python-only comparison script for Qwen evaluating standard HuggingFace Peft adapters instead of using external serving tools like Ollama or vLLM.
**Rationale:** Required a way to rigorously compare precisely the baseline, SFT, and GRPO checkpoint adapter delta directly on the Kaggle T4, while avoiding Unsloth installation friction for inference evaluation and avoiding VL-processor confusion for text-only pipeline runs.
**Tradeoff:** Models are loaded sequentially and unloaded strictly to save VRAM, impacting total evaluation runtime slightly but maximizing safety on 15GB T4 constraints.

---

### 4/25/2026 - GRPO Hardware & Config Decisions

- DECISION: Keep 8-bit quantization for training (VRAM constraint on T4 15.6GB). Export as merged 16-bit for inference. This is the correct tradeoff.
- DECISION: n=4 generations per prompt in GRPO (T4 memory constraint). Would prefer n=8 but not feasible.
- DECISION: SFT and GRPO are now separated into distinct notebook cells so GRPO can be re-run independently without re-running SFT.
- DECISION: Remove EarlyStoppingCallback entirely rather than fight Unsloth 8-bit compatibility. 2 epochs is sufficient overfitting protection with reduced LR.
- PENDING DECISION: GRPO reward collapse fix — options being evaluated: (a) reduce LR 5e-5->1e-5, (b) add cosine LR scheduler, (c) increase warmup steps, (d) soften invalid JSON penalty from -0.3 to -0.1 for near-misses, (e) reduce max_grad_norm 1.0->0.3.
