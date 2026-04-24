# Instructions

## Source of Rules
This file is the source of rules for this workspace.

## Mandatory Workflow
1. Read this file before making any code or config changes.
2. Read Context.md to understand the current project state.
3. After every meaningful step, append an entry to Change_log.md.
4. Prefer small, reversible changes and verify behavior after edits.

## Coding Rules
- Keep changes focused on the requested task.
- Preserve existing behavior unless the task explicitly requires behavior changes.
- Update related documentation when behavior changes.
- Use clear naming for any new files created during experiments or variants.

## Round 2 Priority
- Implement schema drift aware behavior for the data pipeline workflow.
- Prioritize drift detection and adaptation in model decision flow.

---

## Round 2 Submission Checklist

### BLOCKING — automated validation will fail without these

- [x] **Fix inference.py env var compliance (GAP-001)** — DONE 2026-04-24
  - Reads API_BASE_URL, HF_TOKEN, MODEL_NAME from environment variables.
  - FALLBACK_ACTION is now compare_schema (not run_pipeline).
  - handle_drift included in SYSTEM_PROMPT.

- [x] **Write README.md (GAP-002)** — DONE 2026-04-24
  - Covers observation/action spaces, all 4 tasks, reward model, setup, anti-patterns.

- [x] **Prepare training script (GAP-003)** — DONE 2026-04-24
  - training_grpo.ipynb: SFT on gold trajectories then GRPO with live env reward.
  - Kaggle T4 compatible, ~90 min total runtime.
  - Includes evaluation, reward curve plot, schema drift demo, HF Hub push.

- [x] **Deploy HuggingFace Space and verify POST /reset returns HTTP 200 (GAP-005)**
  - IMPORTANT: src/server.py uses WebSocket protocol. The HF Space must also expose
    an HTTP REST endpoint for the validator (POST /reset, POST /step, GET /health).
  - Either add a FastAPI HTTP router alongside the WebSocket, or wrap with a thin REST adapter.

- [x] **Run openenv validate and confirm all checks pass (GAP-006)**
  - Command: pip install openenv-core && openenv validate
  - Fix any yaml schema errors before submission.

### REQUIRED FOR PITCH

- [ ] **Record 2-minute video or mini blog post (GAP-004)**
  - Post on HuggingFace (model card) or YouTube.
  - Core story: base model patches column "spend" that no longer exists after drift.
    Trained model calls compare_schema -> detects rename -> handle_drift(resolve_column_rename).
  - Show the schema drift demo cell output from training_grpo.ipynb.

- [ ] **Prepare 3-minute pitch + 2-minute Q&A**
  - Structure:
    - 0:00-0:30  Problem (production pipeline failures from silent schema renames)
    - 0:30-1:30  Environment design (4 tasks, 11 actions, shaped reward, drift schedule)
    - 1:30-2:15  Schema drift novelty (dynamic mid-episode contract mutation)
    - 2:15-2:45  Training results (SFT -> GRPO reward curves, before/after score table)
    - 2:45-3:00  Live demo (training_grpo.ipynb schema drift demo cell)
  - Scoring weights: Innovation 40%, Storytelling 30%, Reward Improvement 20%, Training 10%.
  - Lead line: "On any given day, 20% of production pipeline failures are silent schema renames
    from upstream APIs. We built an environment where AI learns to detect and recover from
    these with no human intervention."

### NICE-TO-HAVE
- [x] Reward curves from GRPO training showing mean episode reward improving vs. step. (Implemented in all 3 training scripts)
- [x] Side-by-side comparison: before/after score table (easy/medium/hard/hard2) with delta. (Implemented in all 3 training scripts)
- [x] Push trained model to HuggingFace Hub so judges can reproduce. (Implemented in all 3 training scripts)

---
*Note: We have implemented 3 training variants:*
1. `train_grpo_general.ipynb` / `train_grpo.py` (General Model Agnostic)
2. `training_grpo_qwen.ipynb` (Qwen2.5-1.5B-Instruct for Kaggle T4)
3. `training_grpo.ipynb` (Original LLaMA 8B with Gemini trajectories)

---

## Session 2 Fixes (2026-04-24)
- [x] **3 silent environment logic bugs fixed** in `src/environment.py` — `mark_acceptable` override, `add_data_filter` operator validation, `read_data_sample` column guard.
- [x] **NEW inference notebook**: `run_on_kaggle/run_on_kaggle_qwen_1.5b.ipynb` — text-only Qwen2.5-1.5B, MAX_STEPS=100, MAX_TOKENS=1024.
- [x] **All 6 Kaggle notebooks patched**: `importlib` cache flush + defensive `try/except TypeError` env creation to fix the `max_steps` kwarg error.
