# Change Log

## 2026-04-23
- Initialized mandatory workflow files: Instructions.md, Context.md, Change_log.md.
- Added baseline rules and current-state context for Round 2 schema drift work.
- Reviewed current implementation in inference_gemini.py and action model constraints in src/models.py.
- Confirmed environment currently supports compare_schema but not a native handle_drift action.
- Created new Round 2 inference variant: inference_gemini_round2_schema_drift.py (copied from inference_gemini.py).
- Implemented schema-drift upgrade in new inference file:
  - Added schema drift signals in user prompt construction.
  - Added virtual action handle_drift in system instructions.
  - Added action normalization layer mapping handle_drift strategies to supported env actions.
  - Changed fallback behavior from run_pipeline to compare_schema for safer diagnosis.
- Performed syntax validation in Basic_Computer_vision environment. Result: success.
- Completed full Round 2 integration pass:
  - Added native handle_drift action to src/models.py.
  - Added handle_drift dispatcher + strategy handlers in src/environment.py.
  - Added run-indexed drift scheduler (_apply_scheduled_drift) triggered during run_pipeline.
  - Added new hard2 task in src/tasks.py while keeping existing hard task unchanged.
  - Added hard2 drift schedule: column rename -> auth format rotation -> rate limit tightening.
  - Updated task registries in inference.py, inference_gemini.py, and inference_gemini_round2_schema_drift.py.
  - Updated assertion behavior to report missing-column failures explicitly (src/assertions.py).
  - Updated pipeline aggregation to tolerate spend/total_spend in joins (src/pipeline_runner.py).
- Validation completed:
  - py_compile succeeded for all modified Python files.
  - hard2 smoke test: run2 applied drift (spend -> total_spend), handle_drift resolved it, run3 applied auth rotation.
- Implemented OpenEnv stdout logging ([START], [STEP], [END]) across all inference scripts.
  - Redirected all verbose output to sys.stderr.
  - Applied score clipping: min(max(raw_score, 0.01), 0.99).
- Fixed Windows UnicodeEncodeError by removing all emojis. Replaced with text markers ([PASSED], [WARNING]).
- Created Kaggle notebook (run_on_kaggle_LlaMa.ipynb) using LLaMA 3.1 8B Instruct with 4-bit quantization.

## 2026-04-24
- Performed full Round 2 submission audit against hackathon_guide.docx checklist.
- Identified and resolved three blocking delivery gaps in one pass.

### FIX: inference.py — OpenEnv spec compliance (GAP-001 RESOLVED)
- Replaced hardcoded API_BASE_URL/API_KEY/MODEL_NAME with os.getenv() calls per spec.
  - API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
  - API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "MISSING_KEY"
  - MODEL_NAME   = os.getenv("MODEL_NAME", "llama3")
- Changed FALLBACK_ACTION from run_pipeline to compare_schema to avoid blind-fix penalty on fallback.
- Added handle_drift to SYSTEM_PROMPT with all 8 strategies enumerated.
- Added historical_schema to build_user_prompt (was missing vs. Gemini variant).
- Fixed read/fix action counter to include handle_drift as a fix action.
- Removed stale getattr(obs, "last_action_error") pattern (field does not exist in model).
- Cleaned up all dead commented-out code blocks.
- Validated: python -m py_compile inference.py — success.

### NEW: README.md (GAP-002 RESOLVED)
- Created README.md in project root with:
  - Environment overview and Round 2 novelty (dynamic schema drift mid-episode).
  - Full observation space table (all PipelineObservation fields).
  - Full action space table (all 11 actions with key params).
  - Reward model table (per-assertion rewards, bonuses, penalties).
  - Task descriptions for all 4 tasks with faults, assertions, fixes, and baseline scores.
  - Setup instructions: pip install, local server, Docker, openenv validate.
  - Project structure tree.
  - WebSocket protocol spec.
  - Anti-patterns tested table.

### NEW: training_grpo.ipynb (GAP-003 RESOLVED)
- Created training_grpo.ipynb — full two-stage training notebook, Kaggle T4 compatible.
- Stage 1 SFT (~30 min on T4):
  - collect_gold_trajectories() runs env with gold actions on easy/medium tasks.
  - Formats (obs, action) pairs as chat-template strings for Unsloth SFTTrainer.
  - 3 epochs, AdamW-8bit, cosine LR, gradient checkpointing. Saves to sft_checkpoint/.
- Stage 2 GRPO (~60 min on T4):
  - pipeline_reward_fn() executes model actions in live environment per completion.
  - Reward components: env step reward + format bonus (valid JSON +0.3) + drift detection bonus (+0.3).
  - KL coefficient 0.1 against SFT reference to prevent policy collapse.
  - G=4 completions per prompt, hard/hard2 task distribution.
  - Saves to grpo_checkpoint/.
- Evaluation cell: runs all 4 tasks, prints before/after score table with delta column.
- Reward curve cell: plots GRPO training loss and mean episode reward vs. step.
- Schema drift demo cell: shows base model hallucinating old column name vs. trained model calling compare_schema first.
- Hub push cell: pushes merged 16-bit model to HuggingFace Hub.

### REMAINING GAPS (still open before submission)
- GAP-004: Mini blog or 2-minute video not yet recorded.
- GAP-005: HuggingFace Space deployment and POST /reset HTTP 200 not yet verified.
- GAP-006: openenv validate not yet run on openenv.yaml.
