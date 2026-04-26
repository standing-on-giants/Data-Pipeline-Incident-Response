---
title: Data Pipeline Incident Qwen GRPO
emoji: 📊
colorFrom: pink
colorTo: pink
sdk: docker
pinned: false
---

# Data Pipeline Incident Response — OpenEnv Environment

**Meta PyTorch OpenEnv Hackathon · Round 2**  
**Theme: 3.1 — World Modeling: Professional Tasks**  
**Team alphazero — Shashank · Abhinav · Pratham**

---

## Overview

This environment simulates a real-world data engineering incident room. Every day, production data pipelines silently fail because upstream APIs change their payload schemas without warning — columns get renamed, numeric fields arrive as currency strings, deduplication keys shift format, and join keys grow unexpected prefixes.

An AI agent is placed inside a broken pipeline and must diagnose the root cause, apply the correct fix, and verify the pipeline passes — without hallucinating fixes it hasn't confirmed via data inspection.

**Round 2 novel contribution:** Dynamic schema drift injected mid-episode. After the agent applies initial fixes, the upstream API *mutates again* (column rename → auth format rotation → rate limit tightening), forcing the agent to detect and adapt to live contract changes — not just resolve a static fault.

---

## Table of Contents

1. [Environment Spec](#environment-spec)
2. [Observation Space](#observation-space)
3. [Action Space](#action-space)
4. [Reward Model](#reward-model)
5. [Tasks](#tasks)
6. [Server API Reference](#server-api-reference)
   - [HTTP Endpoints](#http-endpoints)
   - [WebSocket Protocol](#websocket-protocol)
7. [Project Structure](#project-structure)
8. [Source Module Reference](#source-module-reference)
9. [Training Pipeline](#training-pipeline)
   - [Stage 1: SFT](#stage-1-supervised-fine-tuning-sft)
   - [Stage 2: GRPO](#stage-2-group-relative-policy-optimization-grpo)
10. [Model Artifacts](#model-artifacts)
11. [Setup & Running](#setup--running)
12. [Key Design Decisions](#key-design-decisions)
13. [Anti-Patterns Tested](#anti-patterns-tested)

---

## Environment Spec

| Property | Value |
|---|---|
| Protocol | OpenEnv WebSocket (`/ws`) + HTTP REST |
| Health endpoint | `GET /health` |
| Action space | 11 discrete typed actions |
| Observation space | Pydantic `PipelineObservation` |
| Episode length | Max 20 steps (hard2: 30 steps) |
| Reward range | [−1.0, +1.5] per step; terminal +1.0 bonus on full pass |
| Tasks | `easy`, `medium`, `hard`, `hard2` |
| Server port | `7860` (default; override with `PORT` env var) |

---

## Observation Space

Every `reset()` and `step()` returns a `PipelineObservation` Pydantic model with these fields:

```python
class PipelineObservation(BaseModel):
    task_id: str
    difficulty: str
    description: str
    step_number: int
    max_steps: int
    dag_structure: List[DAGNode]
    failed_assertions: List[AssertionResult]
    passed_assertions: List[AssertionResult]
    historical_runs: List[HistoricalRun]
    data_sample: Optional[List[Dict[str, Any]]]   # populated after read_data_sample
    current_schema: Optional[Dict[str, str]]       # populated after check_schema
    historical_schema: Optional[Dict[str, str]]    # populated after compare_schema
    schema_diff: Optional[Dict[str, str]]          # new/removed/changed columns
    available_tables: List[str]                    # all legal table names
    last_action_result: str
    actions_taken: List[str]
    pipeline_passed: bool
    alert_sent: bool
```

### Sub-models

```python
class AssertionResult(BaseModel):
    assertion_id: str
    table: str
    assertion_type: str           # not_null | unique | row_count | value_range | type_check
    column: Optional[str]
    expected: str
    actual: str
    passed: bool
    failing_row_count: int

class DAGNode(BaseModel):
    step_id: str
    input_table: str
    output_table: str
    transformation_description: str
    applied_filters: List[str]
    applied_patches: List[str]

class HistoricalRun(BaseModel):
    date: str
    status: str                   # passed | failed
    row_count: int
    duration_s: int
```

---

## Action Space

All actions are submitted as JSON: `{"action_type": "...", "params": {...}}`

| Action | Required Params | Optional Params | Description |
|---|---|---|---|
| `read_data_sample` | `table` | `n_rows` (default 20) | Read rows from a table. **Required before any fix** — blind fixes incur −0.5 penalty. |
| `check_schema` | `table` | — | Inspect current column names and types. Populates `current_schema`. |
| `compare_schema` | `table` | — | Diff current vs historical schema. Populates `schema_diff`. Detects renames/additions/removals. |
| `handle_drift` | `strategy`, `table` | `old_column`, `new_column` | Handle schema/contract drift. See strategies below. |
| `run_quality_assertion` | `assertion_id` | — | Re-run one specific assertion on demand. |
| `add_data_filter` | `step_id`, `filter_condition` | — | Add a WHERE-style row filter, e.g. `user_id IS NOT NULL`. |
| `patch_transformation` | `step_id`, `patch_type`, `column` | `target_type` | Apply a column-level fix. |
| `backfill_partition` | `date` | — | Re-run pipeline for a specific date partition. |
| `alert_upstream_team` | `team`, `issue_description` | — | Escalate to the data source owner. +0.5 if required, −0.2 if unnecessary. |
| `mark_acceptable` | `assertion_id`, `reason` | — | Consciously accept a known data quality issue. −1.0 if still fixable. |
| `run_pipeline` | — | — | Re-execute the full pipeline. Returns new assertion results and reward delta. |

### `handle_drift` Strategies

| Strategy | Purpose |
|---|---|
| `detect` | Identify the type of drift without applying any fix |
| `resolve_column_rename` | Remap `old_column` → `new_column` across the DAG |
| `numeric_format` | Fix currency/percentage string values (e.g. `"$1,234"` → `1234.0`) |
| `null_fill` | Fill nulls introduced by upstream API changes |
| `type_cast` | Cast a column to the correct type |
| `join_key_prefix` | Strip unexpected prefixes from join keys (e.g. `CMP_12345` → `12345`) |
| `filter_invalid` | Drop rows that fail a value constraint |
| `alert_upstream` | Notify the API team about a contract violation |

### `patch_transformation` Patch Types

| Patch Type | Effect |
|---|---|
| `cast_column` | Cast column to `target_type` (float, int, str) |
| `coalesce` | Replace null values with a fallback default |
| `dedup` | Deduplicate rows by the given column |
| `parse_currency` | Strip `$`, `,`, `%` and cast to float |
| `strip_prefix` | Remove a known string prefix from a column |

---

## Reward Model

| Event | Reward |
|---|---|
| Each assertion newly passing after `run_pipeline` | **+0.4** |
| Each assertion newly failing after `run_pipeline` | **−0.5** |
| Correct upstream escalation (required by task) | **+0.5** |
| Unnecessary upstream escalation | **−0.2** |
| `handle_drift(resolve_column_rename)` successful | **+0.2** |
| Fix applied without calling `read_data_sample` or `check_schema` first | **−0.5** |
| `mark_acceptable` on a still-failing, fixable assertion | **−1.0** |
| All assertions passing at episode end (terminal bonus) | **+1.0** |

**Episode score** = `assertions_passed / assertions_total`, clipped to `[0.01, 0.99]`.

---

## Tasks

### `easy`
- **Fault:** 5 of 100 rows in `raw_orders` have null `user_id` (upstream export misfire caused by a new nullable column).
- **Assertions:** `A1` not_null(user_id), `A2` unique(order_id), `A3` row_count(80–110)
- **Optimal fix:** `add_data_filter` → `user_id IS NOT NULL`
- **Baseline (Gemini 2.5 Flash):** ~0.99

### `medium`
- **Fault:** Vendor resent 20 duplicate `order_item_id` rows — inflating row counts and breaking uniqueness.
- **Assertions:** `B1` unique(order_item_id), `B2` not_null(order_id), `B3` row_count(195–205), `B4` value_range(unit_price)
- **Optimal fix:** `patch_transformation` → `dedup(order_item_id)`
- **Baseline (Gemini 2.5 Flash):** ~0.95

### `hard`
- **Fault:** Four cascading failures — Meta Graph API v19 changed `spend`/`impressions` to currency strings with `N/A` values; Conversions API retried events causing duplicates; join key has `CMP_` prefix mismatch. Upstream alert is required.
- **Assertions:** `H1`–`H8` across `clean_insights`, `clean_conversions`, `roas_summary`
- **Optimal fix:** `parse_currency` + `coalesce` on spend → `dedup` on event_id → `strip_prefix` on campaign_id → `alert_upstream_team` → `run_pipeline`
- **Baseline (Gemini 2.5 Flash):** ~0.75

### `hard2`
- **Fault:** Same as `hard`, plus dynamic drift scheduled between `run_pipeline` calls:
  - Run 2: `spend` renamed to `total_spend` in `raw_ads_insights`
  - Run 3: Auth token format rotated to `Bearer-v2`
  - Run 4: Rate limit tightened to 1 call/window
- **Required behavior:** Call `compare_schema` after each `run_pipeline` to detect live drift; use `handle_drift(resolve_column_rename)` before re-patching.
- **Assertions:** Same `H1`–`H8`
- **Baseline (Gemini 2.5 Flash):** ~0.88 (steps exhausted)
- **Baseline (untrained LLaMA 3.1 8B):** ~0.30 (hallucinates column name after drift)

---

## Server API Reference

The server is built on **FastAPI** and serves at `http://0.0.0.0:7860` by default.  
Interactive docs available at `/docs` (Swagger UI) and `/redoc`.

### HTTP Endpoints

#### `GET /`
Returns server status and version.

**Response:**
```json
{"status": "ok", "env": "data-pipeline", "version": "1.1.0"}
```

---

#### `GET /health`
Health check. Used by OpenEnv harness and Docker health-check.

**Response:**
```json
{"status": "healthy"}
```

---

#### `GET /tasks`
List all available task IDs.

**Response:**
```json
{
  "tasks": [
    {"task_id": "easy",   "difficulty": "easy"},
    {"task_id": "medium", "difficulty": "medium"},
    {"task_id": "hard",   "difficulty": "hard"},
    {"task_id": "hard2",  "difficulty": "hard"}
  ]
}
```

---

#### `POST /reset`
Initialize or restart an episode for a given task. Creates a new internal session.

**Request body** (optional; defaults to `easy`):
```json
{"task_id": "hard2"}
```

**Response:**
```json
{
  "session_id": "hard2_140234567890",
  "observation": { ...PipelineObservation... }
}
```

---

#### `POST /step`
Execute one action in the current episode. Accepts two action payload shapes:

**Shape 1 — flat** (action fields at top level):
```json
{
  "session_id": "hard2_140234567890",
  "action_type": "read_data_sample",
  "params": {"table": "raw_ads_insights", "n_rows": 20}
}
```

**Shape 2 — nested** (action inside an `"action"` key):
```json
{
  "session_id": "hard2_140234567890",
  "action": {
    "action_type": "patch_transformation",
    "params": {"step_id": "transform_insights", "patch_type": "dedup", "column": "event_id"}
  }
}
```

> `session_id` is optional if `/reset` was called immediately before. Server tracks the last active session.

**Response:**
```json
{
  "observation": { ...PipelineObservation... },
  "reward": 0.8,
  "done": false,
  "terminated": false,
  "truncated": false
}
```

---

#### `GET /state`
Return the current observation state for the active session without taking a step.

**Response:** Full `PipelineObservation` JSON.

---

### WebSocket Protocol

Connect to `ws://host:7860/ws`. The server handles three message types on the same socket:

#### Reset
```json
{"action": "reset", "task_id": "easy"}
```
**Server response:**
```json
{
  "session_id": "easy_140234567890",
  "observation": { ...PipelineObservation... },
  "reward": 0.0,
  "done": false,
  "terminated": false,
  "truncated": false,
  "info": {"message": "Environment reset"}
}
```

#### Step
```json
{
  "session_id": "easy_140234567890",
  "action": {
    "action_type": "run_pipeline",
    "params": {}
  }
}
```
**Server response:**
```json
{
  "session_id": "easy_140234567890",
  "observation": { ...PipelineObservation... },
  "reward": 1.4,
  "done": true,
  "terminated": true,
  "truncated": false,
  "info": {}
}
```

#### State
```json
{"action": "state", "session_id": "easy_140234567890"}
```
**Server response:** Full current state dict for the session.

#### Error handling
If a message contains an invalid action payload, the server returns:
```json
{"error": "Invalid action payload: <validation detail>"}
```

---

## Project Structure

```
.
├── server/
│   └── app.py                    # FastAPI server — HTTP + WebSocket endpoints
├── src/
│   ├── environment.py            # DataPipelineEnv: reset(), step(), state()
│   ├── models.py                 # Pydantic: PipelineAction, PipelineObservation, StepResult
│   ├── tasks.py                  # Task definitions: easy, medium, hard, hard2
│   ├── assertions.py             # Deterministic assertion checker
│   └── pipeline_runner.py        # DAG execution engine (filters, patches, aggregations)
├── train_grpo/
│   ├── train_grpo_qwen_merged.ipynb   # Primary Kaggle training notebook (Qwen2.5-3B)
│   └── training_grpo_qwen_merged.py   # Script mirror of the notebook (git-visible)
├── inference_qwen_comparison_GRPO_vs_og.py   # Local 3-way comparison: Base vs SFT vs GRPO
├── inference_gemini_round2_schema_drift.py   # Gemini 2.5 Flash inference script
├── inference_qwen3-vl-4b_round2_schema_drift.py  # Qwen3-VL 4B via Ollama
├── inference.py                  # OpenEnv-compliant baseline entrypoint
├── openenv.yaml                  # OpenEnv manifest (name, tasks, action/obs space)
├── pyproject.toml                # Python project metadata
├── Dockerfile                    # Docker image building server/app.py on port 7860
├── requirements.txt
├── mini_blog_huggingface_round2.md   # HuggingFace blog post
├── storytelling_pitch.md         # Hackathon pitch narrative
├── files/
│   └── video_script.md           # 2-minute demo video script
├── Context.md                    # Living context for ongoing session continuity
├── Decisions.md                  # Architectural decision log
└── Change_log.md                 # Full chronological change log
```

---

## Source Module Reference

### `src/environment.py` — `DataPipelineEnv`

The core environment class. Implements the OpenEnv Gym-style interface.

```python
class DataPipelineEnv:
    def reset(task_id: str) -> tuple[PipelineObservation, dict]
    def step(action: PipelineAction) -> StepResult
    def state() -> dict
```

- **`reset()`** — Initialises the pipeline state for a task. Injects the pre-configured faults (null values, duplicates, currency strings, etc.) and returns the initial observation.
- **`step()`** — Executes one action. Computes the reward, updates the DAG state (filters/patches), runs the drift scheduler if `run_pipeline` is called on an eligible run index, and returns `StepResult`.
- **Blind-fix detection** — The environment tracks whether `read_data_sample` or `check_schema` has been called before any patch/filter is applied. If not, it deducts −0.5.
- **Run-indexed drift** (hard2) — After each `run_pipeline`, if the run count matches a scheduled drift run, the upstream "API" mutates (`schema_rename`, `auth_format_rotation`, `rate_limit_tighten`). The agent must call `compare_schema` to see the mutation.

### `src/models.py` — Pydantic Schema

All models are Pydantic v2 `BaseModel` with full JSON serialisation:
- `PipelineObservation` — agent's view of the world
- `PipelineAction` — typed `Literal` union of 11 action strings
- `StepResult` — wraps observation + reward + done flags + info dict
- `AssertionResult`, `DAGNode`, `HistoricalRun` — sub-models

### `src/tasks.py` — Task Definitions

Defines the `TASKS` registry — a dict mapping task IDs to `TaskConfig` objects describing:
- Initial fault setup (which rows to corrupt, which schema mutations to apply)
- Assertion suite
- Drift schedule (for hard2)

### `src/assertions.py` — Deterministic Assertion Checker

Stateless function `run_assertions(pipeline_output, task_config) -> List[AssertionResult]`.  
Supports: `not_null`, `unique`, `row_count`, `value_range`, `type_check`.

### `src/pipeline_runner.py` — DAG Execution Engine

Applies all configured `add_data_filter` and `patch_transformation` operations to the raw data tables and runs DAG step aggregations. Returns `clean_*` output tables for assertion checking.

---

## Training Pipeline

Model: **Qwen/Qwen2.5-3B-Instruct** with LoRA adapters (r=32, α=32)  
Hardware: Kaggle T4 GPU (16 GB VRAM)  
Stack: Unsloth + HuggingFace TRL + bitsandbytes 8-bit  

All adapters are published as LoRA-only (~200 MB) — **no merged base weights**.

### Stage 1: Supervised Fine-Tuning (SFT)

**Goal:** Enforce output format discipline and teach the basic diagnostic workflow.

**Data collection:** 40 episodes × 4 tasks × gold action sequences = ~1,600 (obs, action) pairs.  
Gold trajectories are collected by replaying the `GOLD_ACTIONS` map through the live environment.

```python
GOLD_ACTIONS = {
    'easy':  [read_data_sample, add_data_filter, run_pipeline],
    'medium': [read_data_sample, patch_transformation(dedup), run_pipeline],
    'hard':   [read_data_sample, compare_schema, parse_currency,
               coalesce, dedup, strip_prefix, alert_upstream, run_pipeline],
    'hard2':  [...hard actions..., run_pipeline, compare_schema,
               handle_drift(resolve_column_rename), run_pipeline],
}
```

**Training config:**
| Param | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Quantization | 8-bit (bitsandbytes) |
| LoRA rank | 32, alpha 32 |
| Epochs | 4 |
| Learning rate | `5e-5` with cosine decay |
| Batch size | 1 with `gradient_accumulation_steps=8` |
| Optimizer | `adamw_8bit` |
| Sequence length | 2048 |
| Eval strategy | every 10 steps |
| Save strategy | every 20 steps, `save_total_limit=3` |

**Output adapter:** `Abhinav-hf/qwen-sft-lora-adapter`

### Stage 2: Group Relative Policy Optimization (GRPO)

**Goal:** Train the agent to maximise shaped environment rewards on hard/hard2 tasks.

The agent generates actions, the live `DataPipelineEnv` executes them and returns rewards, and the policy updates toward higher-reward behaviors via REINFORCE-style policy gradient.

**Training config:**
| Param | Value |
|---|---|
| Epochs | 1 |
| Episodes per task | 20 (4 tasks × 20 = 80 episodes total) |
| Learning rate | `1e-5` |
| Cosine warmup | 20 steps |
| `max_grad_norm` | 0.3 |
| Optimizer | `PagedAdamW8bit` |
| Max episode steps | 6 |
| Max new tokens | 100 |

**Reward signal per GRPO step:**
```python
step_reward = env_reward                              # base environment reward
if n_passed_after > n_passed_before:
    step_reward += 0.15 * (n_passed_after - n_passed_before)  # assertion progress bonus
elif step_reward <= 0:
    step_reward -= 0.1                                # no progress penalty
step_reward -= 0.02                                   # efficiency penalty (encourages brevity)

if obs.pipeline_passed:
    step_reward += 1.0                                # terminal success bonus
```

**Stability features:**
- **Abort gate:** If >25/30 episodes at step 30 have parse failures, GRPO reverts to SFT weights to prevent reward collapse.
- **Best checkpoint tracking:** Saves the adapter with the highest rolling-10 episode reward, not just the final epoch.
- **Partial-credit JSON reward:** Near-miss JSON (valid braces but unparseable) receives −0.1 instead of −0.3.
- **SFT diagnostic run:** Before GRPO begins, runs 4 evaluation episodes (one per task) and prints parse rate. Stops if model is already broken.

**Output adapter:** `Abhinav-hf/qwen-grpo-lora-adapter` (final) and  
`Abhinav-hf/qwen-grpo-best-lora-adapter` (best rolling reward checkpoint)

---

## Model Artifacts

| Adapter | HuggingFace Repo | Contents |
|---|---|---|
| SFT | [`Abhinav-hf/qwen-sft-lora-adapter`](https://huggingface.co/Abhinav-hf/qwen-sft-lora-adapter) | `adapter_config.json`, `adapter_model.safetensors`, tokenizer files |
| GRPO | [`Abhinav-hf/qwen-grpo-lora-adapter`](https://huggingface.co/Abhinav-hf/qwen-grpo-lora-adapter) | Same structure |
| GRPO best | [`Abhinav-hf/qwen-grpo-best-lora-adapter`](https://huggingface.co/Abhinav-hf/qwen-grpo-best-lora-adapter) | Best rolling-reward checkpoint |

**Loading any adapter:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)
model = PeftModel.from_pretrained(base, "Abhinav-hf/qwen-grpo-lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("Abhinav-hf/qwen-grpo-lora-adapter")
```

**Running the 3-way comparison (Base vs SFT vs GRPO):**
```bash
conda activate Basic_Computer_vision

# All three models, all tasks
python inference_qwen_comparison_GRPO_vs_og.py --models base sft grpo

# GRPO only, on hard/hard2 tasks
python inference_qwen_comparison_GRPO_vs_og.py --models grpo --tasks hard hard2

# With 8-bit quantization (fits on T4 16 GB)
python inference_qwen_comparison_GRPO_vs_og.py --use-8bit --models grpo
```

CLI flags:
```
--models   base sft grpo        Which models to evaluate (default: base grpo)
--tasks    easy medium hard hard2   Which tasks to run (default: all)
--steps    N                    Max steps per episode (default: 25)
--use-8bit                      Load base model in 8-bit (saves ~2 GB VRAM)
```

---

## Setup & Running

### Requirements

```bash
pip install fastapi uvicorn pydantic pandas numpy openai python-dotenv
```

### Run the server locally

```bash
python -m server.app

# Or with explicit port
PORT=8001 python -m server.app
```

### Run inference scripts locally

```bash
# Baseline inference
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
python inference.py --task easy

# Gemini schema-drift variant
export GEMINI_API_KEY="..."
python inference_gemini_round2_schema_drift.py --task hard2
```

### Docker

```bash
docker build -t data-pipeline-env .
docker run -p 7860:7860 data-pipeline-env
```

The `Dockerfile` runs `uvicorn server.app:app --host 0.0.0.0 --port 7860`.

### OpenEnv validation

```bash
pip install openenv-core
openenv validate
```

---

## Key Design Decisions

See [`Decisions.md`](Decisions.md) for the full architectural decision log. Key choices:

- **`handle_drift` as a native action** — not a normalization layer. Gives the RL agent the correct credit signal for drift-handling behavior.
- **Run-indexed drift schedule** — triggered inside `run_pipeline` so drift is invisible at trigger time. Mirrors realistic production API behavior.
- **Fallback action = `compare_schema`** — not `run_pipeline`, to avoid compounding blind-fix penalties on fallback.
- **Blind-fix penalty (−0.5)** — agent must call `read_data_sample` or `check_schema` before any patch or filter. Eliminates hallucinated fixes.
- **`mark_acceptable` penalty (−1.0)** — heavy punishment for sweeping real failures under the rug.
- **LoRA-only saves** — `model.save_pretrained()` / `model.push_to_hub()` instead of `save_pretrained_merged`. Keeps HF repos at ~200 MB vs ~6 GB and avoids the slow startup of baking deltas into base weights.

---

## Anti-Patterns Tested

| Anti-pattern | Penalty | Why it matters |
|---|---|---|
| Applying any filter/patch without reading data first | −0.5 | Forces real diagnosis, not guessing |
| Marking a failing, fixable assertion as acceptable | −1.0 | Prevents lazy "it's fine" behavior |
| Unnecessary upstream escalation | −0.2 | Escalation should be earned by evidence |
| Patching without calling `compare_schema` after drift | 0 progress | Tests continuous drift detection |
| Repeating the same failing action in a loop | Cumulative −0.1/step | Breaks action loops |
