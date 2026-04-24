# Data Pipeline Incident Response — OpenEnv Environment

**Meta PyTorch OpenEnv Hackathon · Round 2**
**Theme: 3.1 — World Modeling: Professional Tasks**

---

## Overview

This environment simulates a real-world data engineering incident room. Every day, production data pipelines silently fail because upstream APIs change their payload schemas without warning — columns get renamed, numeric fields arrive as currency strings, deduplication keys shift format, and join keys grow unexpected prefixes.

An AI agent is placed inside a broken pipeline and must diagnose the root cause, apply the correct fix, and verify the pipeline passes — without hallucinating fixes it hasn't confirmed via data inspection.

**The novel contribution for Round 2:** dynamic schema drift injected mid-episode. After the agent applies initial fixes, the upstream API *mutates again* (column rename → auth format rotation → rate limit tightening), forcing the agent to detect and adapt to live contract changes — not just resolve a static fault.

---

## Environment Spec

| Property | Value |
|---|---|
| Protocol | OpenEnv WebSocket (`/ws`) + HTTP health (`/health`) |
| Action space | 11 discrete typed actions |
| Observation space | Pydantic `PipelineObservation` (see below) |
| Episode length | Max 20 steps |
| Reward range | [−1.0, +1.5] per step; terminal +1.0 bonus on full pass |
| Tasks | `easy`, `medium`, `hard`, `hard2` |

---

## Observation Space

Every `step()` and `reset()` returns a `PipelineObservation` with these fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | str | Task identifier |
| `difficulty` | str | easy / medium / hard |
| `description` | str | Human-readable problem description |
| `step_number` | int | Current step in episode |
| `max_steps` | int | Episode length limit (20) |
| `dag_structure` | list[DAGNode] | Pipeline steps with applied filters/patches |
| `failed_assertions` | list[AssertionResult] | Assertions currently failing |
| `passed_assertions` | list[AssertionResult] | Assertions currently passing |
| `historical_runs` | list[HistoricalRun] | Last 3 run records (date, status, row count) |
| `data_sample` | list[dict] \| None | Populated after `read_data_sample` |
| `current_schema` | dict \| None | Populated after `check_schema` |
| `historical_schema` | dict \| None | Populated after `compare_schema` |
| `schema_diff` | dict \| None | new/removed/changed columns after `compare_schema` |
| `last_action_result` | str | Natural language result of last action |
| `actions_taken` | list[str] | History of actions this episode |
| `pipeline_passed` | bool | True when all assertions pass |
| `alert_sent` | bool | True after `alert_upstream_team` was called |

---

## Action Space

| Action | Key Params | Description |
|---|---|---|
| `read_data_sample` | `table`, `n_rows` | Read rows from a table. Required before any fix — blind fixes incur −0.5 penalty. |
| `check_schema` | `table` | Inspect current column names and types. |
| `compare_schema` | `table` | Diff current schema against historical. Surfaces renames/additions/removals. |
| `handle_drift` | `strategy`, `table`, `old_column`, `new_column` | Handle schema/contract drift. Strategies: `detect`, `resolve_column_rename`, `numeric_format`, `null_fill`, `type_cast`, `join_key_prefix`, `filter_invalid`, `alert_upstream`. |
| `run_quality_assertion` | `assertion_id` | Re-run a specific assertion on demand. |
| `add_data_filter` | `step_id`, `filter_condition` | Add a WHERE-style row filter to a DAG step (e.g. `user_id IS NOT NULL`). |
| `patch_transformation` | `step_id`, `patch_type`, `column` | Apply a column-level fix. patch_types: `cast_column`, `coalesce`, `dedup`, `parse_currency`, `strip_prefix`. |
| `backfill_partition` | `date` | Re-run pipeline for a specific date partition. |
| `alert_upstream_team` | `team`, `issue_description` | Escalate to the data source owner. Rewarded +0.5 when required, penalised −0.2 if unnecessary. |
| `mark_acceptable` | `assertion_id`, `reason` | Consciously accept a known data quality issue. Penalised −1.0 if the assertion is still failing and fixable. |
| `run_pipeline` | — | Re-execute full pipeline. Returns new assertion results and reward delta. |

---

## Reward Model

| Event | Reward |
|---|---|
| Each assertion newly passing after `run_pipeline` | +0.4 |
| Each assertion newly failing after `run_pipeline` | −0.5 |
| Correct upstream escalation (required task) | +0.5 |
| Unnecessary escalation | −0.2 |
| `handle_drift(resolve_column_rename)` successful | +0.2 |
| Fix applied without reading data first (blind fix) | −0.5 |
| `mark_acceptable` on a still-failing assertion | −1.0 |
| All assertions passing (terminal bonus) | +1.0 |

Final episode score = `assertions_passed / assertions_total`, clipped to [0.01, 0.99].

---

## Tasks

### easy
**Fault:** 5 of 100 rows in `raw_orders` have null `user_id` (upstream export misfire caused by a new nullable column).
**Assertions:** A1 not_null(user_id), A2 unique(order_id), A3 row_count(80–110)
**Fix:** `add_data_filter` → `user_id IS NOT NULL`
**Baseline score (Gemini 2.5 Flash):** ~0.99

### medium
**Fault:** Vendor resent 20 duplicate `order_item_id` rows, inflating row counts and breaking uniqueness.
**Assertions:** B1 unique(order_item_id), B2 not_null(order_id), B3 row_count(195–205), B4 value_range(unit_price)
**Fix:** `patch_transformation` → `dedup(order_item_id)`
**Baseline score (Gemini 2.5 Flash):** ~0.95

### hard
**Fault:** Four simultaneous faults — Meta Graph API v19 changed `spend`/`impressions` to currency strings with N/A values; Conversions API retried events causing duplicates; join key has `CMP_` prefix mismatch. Upstream alert is required.
**Assertions:** H1–H8 (8 assertions across `clean_insights`, `clean_conversions`, `roas_summary`)
**Fix:** Multiple patches + dedup + strip_prefix + cast + alert_upstream
**Baseline score (Gemini 2.5 Flash):** ~0.75 (TBD — hard task not yet solved by base model)

### hard2
**Fault:** Same as `hard` plus a dynamic drift schedule applied between `run_pipeline` calls:
- Run 2: `spend` renamed to `total_spend` in `raw_ads_insights`
- Run 3: Auth token format rotated to `Bearer-v2`
- Run 4: Rate limit tightened to 1 call/window

Agent must call `compare_schema` after each run_pipeline to detect live drift and use `handle_drift(resolve_column_rename)` before patching.
**Assertions:** Same H1–H8
**Baseline score (Gemini 2.5 Flash):** ~0.88 (max_steps reached)
**Baseline score (untrained LLaMA 3.1 8B):** ~0.30 (hallucinates column name after drift)

---

## Setup

### Requirements

```bash
pip install fastapi uvicorn pydantic pandas numpy openai python-dotenv
```

### Run locally

```bash
# Start the WebSocket server
python -m src.server

# Run baseline inference (reads API_BASE_URL, MODEL_NAME, HF_TOKEN from env)
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
python inference.py --task easy

# Run all tasks
python inference.py --task all

# Run with Gemini (schema-drift aware variant)
export GEMINI_API_KEY="..."
python inference_gemini_round2_schema_drift.py --task hard2
```

### Docker

```bash
docker build -t data-pipeline-env .
docker run -p 8001:8001 data-pipeline-env
```

### OpenEnv validation

```bash
pip install openenv-core
openenv validate
```

---

## Project Structure

```
.
├── inference.py                              # OpenEnv-compliant baseline (required entrypoint)
├── inference_gemini_round2_schema_drift.py   # Gemini 2.5 Flash variant (schema-drift aware)
├── inference_qwen3-vl-4b_round2_schema_drift.py  # Qwen3-VL 4B via Ollama
├── run_on_kaggle_LlaMa.ipynb                 # Kaggle inference notebook (LLaMA 3.1 8B, 4-bit)
├── openenv.yaml                              # OpenEnv spec metadata
├── Dockerfile
├── src/
│   ├── environment.py    # DataPipelineEnv (reset / step / state)
│   ├── models.py         # Pydantic: PipelineAction, PipelineObservation, StepResult
│   ├── tasks.py          # Task definitions: easy, medium, hard, hard2
│   ├── assertions.py     # Deterministic assertion checker
│   ├── pipeline_runner.py # DAG execution engine (filters, patches, aggregations)
│   └── server.py         # FastAPI WebSocket server
├── Context.md
├── Decisions.md
├── Change_log.md
└── Instructions.md
```

---

## WebSocket Protocol

```
Client → Server: {"action": "reset", "task_id": "easy|medium|hard|hard2"}
Client → Server: {"action": {"action_type": "...", "params": {...}}}
Client → Server: {"action": "state"}

Server → Client: {"observation": {...}, "reward": 0.0, "done": false, "info": {...}}
```

---

## Key Design Decisions

See `Decisions.md` for the full architectural decision log. Key choices:

- **`handle_drift` as a native action** (not a normalization layer) for correct RL credit assignment
- **Run-indexed drift schedule** triggered inside `run_pipeline` so drift is invisible at trigger time (realistic production setting)
- **Fallback action = `compare_schema`** instead of `run_pipeline` to avoid blind-fix penalties
- **Blind-fix penalty (−0.5)**: agent must call `read_data_sample` or `check_schema` before patching — discourages hallucinated fixes
- **`mark_acceptable` anti-pattern (−1.0)**: heavy penalty for sweeping real failures under the rug

---

## Training Approach

Training uses a two-stage SFT → GRPO pipeline:

1. **SFT**: Collect ~50–100 successful trajectories via Gemini 2.5 Flash on easy/medium tasks. Fine-tune LLaMA 3.1 8B with Unsloth (4-bit quantization, Kaggle T4 compatible).
2. **GRPO**: Fine-tune on hard/hard2 tasks with shaped environment reward. KL penalty (~0.1 coefficient) against SFT reference prevents policy collapse.

See `run_on_kaggle_LlaMa.ipynb` for the training notebook.

---

## Anti-Patterns Tested

| Anti-pattern | Penalty | Why it matters |
|---|---|---|
| Applying a filter/patch without reading data | −0.5 | Forces real diagnosis, not guessing |
| Marking a failing assertion as acceptable | −1.0 | Prevents lazy "it's fine" behaviour |
| Unnecessary upstream escalation | −0.2 | Escalation should be earned |
| Patching before calling compare_schema after drift | 0 fix progress | Tests drift detection, not just patching |
