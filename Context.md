# Context

## Project
Meta PyTorch OpenEnv Hackathon — Round 2 Grand Finale
**On-site: 25-26 April 2026, Scaler School of Technology, Bangalore**

## Project: Data Pipeline Incident Response (Project 3)

### Why This Project
- Strongest training delta available: Gemini 2.5 Pro does NOT solve hard2, so real improvement is possible.
- Clean shaped reward signal compatible with GRPO (diagnostic actions, blind-fix penalty, escalation bonus).
- Dynamic schema drift as novel angle: matches Patronus sub-theme (data contracts, schema evolution).
- Theme 3.1 fit: Professional Tasks (real data engineering incidents).

### Theme Alignment
- Primary: Theme 3.1 — World Modeling: Professional Tasks
- Secondary (via schema drift): Patronus sub-theme — data contracts, schema evolution, API reliability

---

## Current Workspace State

### Completed
- Full OpenEnv-compliant environment (src/environment.py, src/models.py, src/assertions.py, src/pipeline_runner.py).
- 4 tasks: easy, medium, hard, hard2.
- Native handle_drift action in models.py and environment.py.
- Run-indexed drift scheduler (_apply_scheduled_drift) triggered inside run_pipeline.
- hard2 drift schedule: column rename (run 2) -> auth rotation (run 3) -> rate limit (run 4).
- OpenEnv stdout logging ([START], [STEP], [END]) in all inference scripts.
- Score clipping min(max(score, 0.01), 0.99) for OpenEnv compliance.
- Terminal sanitized (no emojis) for Windows cross-platform stability.
- WebSocket server (src/server.py).
- inference_gemini_round2_schema_drift.py — Gemini 2.5 Flash schema-drift variant.
- inference_qwen3-vl-4b_round2_schema_drift.py — Qwen3-VL 4B via Ollama.

### Completed on 2026-04-24 (new deliverables)
- inference.py FIXED: now reads API_BASE_URL, HF_TOKEN, MODEL_NAME from env vars per spec.
  - FALLBACK_ACTION changed from run_pipeline to compare_schema.
  - handle_drift added to SYSTEM_PROMPT with all 8 strategies.
  - historical_schema and schema_diff surfaced in prompt builder.
  - All dead commented code removed.
- README.md CREATED: full submission README with obs/action spaces, task specs, reward model, setup.
- 3 GRPO Training Variants Implemented:
  1. `train_grpo.py` / `train_grpo_general.ipynb`: General Model-Agnostic CLI script and notebook (works with any HF model).
  2. `training_grpo_qwen.ipynb`: Optimized for Qwen2.5-1.5B-Instruct (4-bit, LoRA r=32, `<think>` tag stripping) on Kaggle T4.
  3. `training_grpo.ipynb`: LLaMA 8B notebook upgraded with Gemini API trajectory collector and loop penalty.
  - All 3 scripts include: SFT stage, GRPO stage with shaped environment reward (format bonus, drift bonus, loop penalty), reward curve plotting, evaluation comparison table, and Hugging Face Hub push logic.

### Completed on 2026-04-24 (Session 2 — Final Bug Sweep)
- **3 silent environment bugs fixed** in `src/environment.py`:
  - `mark_acceptable` now actually overrides failing assertions to `passed=True` in `_run_all_assertions`. Reward corrected to `+0.1` (was `-1.0`).
  - `add_data_filter` now validates SQL operator syntax. Unsupported operators (e.g. `==`) return `-0.1` and a clear error message instead of silently doing nothing.
  - `read_data_sample` now rejects filter requests on non-existent columns with `-0.1` instead of silently returning unfiltered data.
- **NEW**: `run_on_kaggle/run_on_kaggle_qwen_1.5b.ipynb` — clean text-only Qwen2.5-1.5B-Instruct notebook for Kaggle T4. `MAX_STEPS=100`, `MAX_TOKENS=1024`. VRAM: ~2.4 GB.
- **All 6 Kaggle notebooks patched**: `importlib` module cache flush in clone cell + defensive `try/except TypeError` around env creation.
- Model default updated: `Qwen/Qwen2.5-3B-Instruct` → `Qwen/Qwen2.5-1.5B-Instruct` across training scripts.

### Still Open (must complete before submission day)
- GAP-004: Mini blog or 2-minute video (HuggingFace or YouTube).

---

## Algorithm Plan
- Stage 1 SFT: collect ~50-100 successful trajectories via Gemini on easy/medium. Fine-tune LLaMA 3.1 8B with Unsloth (4-bit).
- Stage 2 GRPO: hard/hard2 tasks, G=4 completions, env reward + format bonus + drift detection bonus. KL=0.1 against SFT reference.
- Expected training time on Kaggle T4: ~90 min total.
- Model to push: standing-on-giants/data-pipeline-incident-llama-grpo

## Baseline Scores (reference)
| Task   | Gemini 2.5 Flash | LLaMA 3.1 8B (untrained) |
|--------|-----------------|--------------------------|
| easy   | ~0.99           | ~0.70                    |
| medium | ~0.95           | ~0.55                    |
| hard   | ~0.75           | ~0.30                    |
| hard2  | ~0.88*          | ~0.30                    |

*Gemini hits max_steps on hard2 without fully passing — real training delta available.
