---
title: Data Pipeline Incident Response — OpenEnv
colorFrom: red
colorTo: orange
sdk: docker
app_port: 7860
tags:
  - openenv
  - data-engineering
  - rl-environment
  - agentic
license: mit
---

# Data Pipeline Incident Response — OpenEnv

An RL environment where an agent acts as an on-call Data Engineer, diagnosing broken pipelines, investigating data quality issues, and applying root-cause fixes. 

---

## Motivation

Data teams deal with broken pipelines daily. A vendor silently changes an API export format, null values suddenly appear in critical source tables, or duplicate rows inflate financial aggregations downstream. Fixing these issues requires more than just code generation — it requires **investigating data dynamically**. 

In this environment, the agent receives an alert with failing assertions in a DAG. It must query the data (`read_data_sample`, `check_schema`) to figure out what went wrong, apply the correct transformation patch (`add_data_filter`, `patch_transformation`), verify the fix by re-running the pipeline, and know when to escalate genuinely corrupted data upstream. 

The environment benchmarks agentic LLMs on tasks that require **data-driven reasoning**, **iterative debugging**, and making the distinction between locally fixable bugs and upstream data corruption.

---

## Project Structure

```
data_pipeline_env/
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── new_inference.py      Agent loop, prompt builder, action parser (local use)
└── src/
    ├── __init__.py
    ├── environment.py    OpenEnv-compliant env (reset / step / state)
    ├── pipeline_runner.py Simulates DAG execution and data transformations 
    ├── assertions.py      Deterministic data quality grader
    ├── tasks.py          Task definitions (DAGs, faults, schemas)
    ├── models.py         Pydantic models (PipelineAction, PipelineObservation, …)
    └── server.py         FastAPI REST wrapper
```
# Data Pipeline Incident Response — OpenEnv

An RL environment where an agent diagnoses and fixes broken data pipelines.

**Real-world task**: Every data team deals with pipeline failures overnight. The agent
acts as an on-call data engineer: it reads failing assertion reports, inspects raw data,
identifies root causes, and applies fixes — exactly what a human would do at 3am.

---

## Quick Start

```bash
docker build -t data-pipeline-env .
docker run -p 7860:7860 data-pipeline-env
```

### Local (Python conda/venv)

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference against all 3 tasks (requires API key)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_token_here"

Push this repo to a Hugging Face Space with `sdk: docker` in the README frontmatter. The Space builds and serves automatically.

```bash
git clone https://huggingface.co/spaces/<your-org>/data-pipeline-incident-response
cp -r . data-pipeline-incident-response/
cd data-pipeline-incident-response
git add . && git commit -m "initial" && git push
python inference.py
python inference.py --task easy    # single task
python inference.py --task hard --steps 25
```

---

## Environment Overview

These are only needed when running `new_inference.py` locally. The Space itself has no model dependency — it only serves the environment API.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes (local) | — | API key for Gemini inference |
| `MODEL_NAME` | No | `gemini-2.5-flash` | Model override |
| `MAX_STEPS` | No | `20` | Max steps per episode |
| `TEMPERATURE` | No | `0.1` | Sampling temperature |
| `MAX_TOKENS` | No | `1024` | Max tokens per LLM call |

---

## Usage

### REST API

```bash
# Health check
curl http://localhost:7860/health

# List tasks
curl http://localhost:7860/tasks

# Start a session
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "max_steps": 20}'
# → {"session_id": "easy_...", "observation": {...}}

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<session_id from reset>",
    "action": {
      "action_type": "read_data_sample",
      "params": {"table": "raw_orders", "n_rows": 20}
    }
  }'

# Get session state
curl "http://localhost:7860/state?session_id=<session_id>"
```

### Local agent runner

```bash
pip install openai python-dotenv   # in addition to requirements.txt
# Set your GEMINI_API_KEY in the .env file

python new_inference.py              # all three tasks
python new_inference.py --task easy
python new_inference.py --task hard --steps 25
python new_inference.py --quiet      # suppress per-step output
```

---

## Environment Description

```python
from src.environment import DataPipelineEnv
from src.models import PipelineAction

env = DataPipelineEnv(task_id="easy")    # "easy" | "medium" | "hard"
obs = env.reset()                        # → PipelineObservation
result = env.step(action)                # → StepResult
state = env.state()                      # → Dict
```

Episode ends when all assertions pass (`pipeline_passed=True`) or the step budget is exhausted.

---

## Observation Space

`PipelineObservation` fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | `"easy"`, `"medium"`, or `"hard"` |
| `difficulty` | `str` | Human-readable difficulty label |
| `description` | `str` | Task context (what broke) |
| `max_steps` | `int` | Step budget |
| `dag_structure` | `List[DAGNode]` | The pipeline graph (steps + schemas) |
| `failed_assertions` | `List[AssertionResult]` | Current failing Data Quality checks |
| `passed_assertions` | `List[AssertionResult]` | Current passing Data Quality checks |
| `historical_runs` | `List[HistoricalRun]` | Pipeline history spanning several days |
| `data_sample` | `Optional[List[Dict]]` | Populated when `read_data_sample` is called |
| `current_schema` | `Optional[Dict]` | Populated when `check_schema` is called |
| `historical_schema` | `Optional[Dict]` | Populated when `compare_schema` is called |
| `actions_taken` | `List[str]` | Recent action history |
| `last_action_result` | `str` | Textual result of the last action |
| `step_number` | `int` | Current step number |
| `pipeline_passed` | `bool` | `True` when all assertions pass |
### The Problem

A data pipeline runs nightly. It:
1. **Extracts** data from upstream sources (Salesforce, vendors, CRMs)
2. **Transforms** it (cleans, joins, aggregates)
3. **Loads** it into downstream tables used by dashboards and ML models

**Data quality assertions** check things like:
- `user_id IS NOT NULL`
- `order_item_id is UNIQUE`
- `revenue is numeric and in [0, 1,000,000]`

When assertions fail, the pipeline halts. The agent's job is to figure out why and fix it.

---

## Tasks

| Task   | Fault                              | Tables | Assertions | Difficulty |
|--------|-------------------------------------|--------|------------|------------|
| easy   | 5 null user_ids from upstream       | 2      | 3          | Easy       |
| medium | 20 duplicate order_item_ids         | 3      | 4          | Medium     |
| hard   | Revenue format change + N/A rows   | 4      | 6          | Hard       |

---

## Action Space

| Action | Params | Purpose |
|---|---|---|
| `read_data_sample` | `table`, `n_rows` | Diagnose by inspecting actual rows |
| `check_schema` | `table` | Diagnose by checking current column dtypes |
| `compare_schema` | `table` | Diff current vs historical schema to spot drift |
| `add_data_filter` | `step_id`, `filter_condition` | Fix: drop bad records (e.g., `user_id IS NOT NULL`) |
| `patch_transformation` | `step_id`, `patch_type`, `column` | Fix: clean data in place (`coalesce`, `dedup`, `parse_currency`) |
| `run_pipeline` | *(none)* | Recompiles and triggers DAG, generating new assertion states |
| `alert_upstream_team` | `team`, `issue_description` | Correctly escalate unfixable upstream corruption |
| `mark_acceptable` | `assertion_id`, `reason` | *(Anti-pattern)* Mark a failing assertion as ignored |

**Rewards:** Checking data is mildly penalized for spamming (`-0.05`), shooting blind (adding filters without reading data) is heavily penalized (`-0.5`), correctly escalating unfixable data gains `+0.5`, and fixing the pipeline gains points based on new assertions passing vs failing. Passing the full pipeline gives a `+1.0` bonus.

---

## Task Descriptions

### easy — Null Values

| Issue | Details |
|---|---|
| **Fault** | Upstream table added a `discount_code` column, accidentally null-ing out `user_id` for 5 random rows. |
| **Symptom** | `NOT NULL` assertion failing on `orders_clean.user_id`. |
| **Solution** | `add_data_filter` (`user_id IS NOT NULL`) |
| Action                 | Description                                          |
|------------------------|------------------------------------------------------|
| `read_data_sample`     | See the first N rows of a table                      |
| `check_schema`         | See current column types                             |
| `compare_schema`       | Diff current schema vs historical                    |
| `run_quality_assertion`| Re-run a specific assertion                          |
| `add_data_filter`      | Add a WHERE clause to a pipeline step                |
| `patch_transformation` | Apply a column fix (cast, dedup, parse_currency)     |
| `backfill_partition`   | Re-run pipeline for a specific date                  |
| `alert_upstream_team`  | Escalate unfixable issues to the data source owner   |
| `mark_acceptable`      | Accept a known issue (penalised if wrong)            |
| `run_pipeline`         | Re-execute full pipeline and check all assertions    |

---

## Reward Design

| Event                                             | Reward    |
|---------------------------------------------------|-----------|
| Assertion goes from failing → passing (per run)   | +0.4      |
| Assertion goes from passing → failing (per run)   | −0.5      |
| All assertions pass (terminal)                    | +1.0      |
| Applying fix without reading data first           | −0.5      |
| Correct upstream alert (hard task)                | +0.5      |
| `mark_acceptable` on genuinely failing assertion  | −1.0      |

---

## Running the Server (Docker)

```bash
# Build
docker build -t data-pipeline-env .

# Run
docker run -p 8001:8001 data-pipeline-env

# Health check
curl http://localhost:8001/health

# WebSocket endpoint: ws://localhost:8001/ws
```

### medium — Duplicated Rows

| Issue | Details |
|---|---|
| **Fault** | Vendor sent 20 duplicate order items for yesterday. |
| **Symptom** | Unique constraint fails on `order_item_id`, and `order_summary` row counts are inflated. |
| **Solution** | `patch_transformation` (type=`dedup`, column=`order_item_id`) on the `transform_items` step. |

---

### hard — Meta Ads & Conversions Pipeline (Cascading Failures)

| Issue | Details |
|---|---|
| **Fault 1** | Graph API v19.0 changed `spend` to `"$1,234.56"` strings. ~10 rows have `"N/A"` from API outage. |
| **Fault 2** | `impressions` has `"N/A"` values; coalescing to 0 causes CTR divide-by-zero (Inf/NaN). |
| **Fault 3** | Conversions API retried failed payloads → ~37 duplicate `event_id` rows (15%). |
| **Fault 4** | `campaign_id` in conversions has `"CMP_"` prefix; insights uses int → ROAS join drops 90%+ of rows. |
| **Symptom** | 8 assertions fail across 3 output tables: type/range on `spend`, range on `ctr`, unique/row_count on conversions, row_count/range on `roas_summary`. |
| **Solution** | Sequential: `parse_currency` + `coalesce` on `spend`, `parse_currency` + `coalesce(1)` on `impressions`, `dedup` on `event_id`, `strip_prefix` + `cast_column` on conversions `campaign_id`, + `alert_upstream_team(meta_ads_api_team)`. |

---

## Baseline Scores

Model: `gemini-2.5-pro`. `MAX_STEPS=20`, `TEMPERATURE=0.1`.

| Task | Score | Reward | Steps | Assertions Passed | Passed |
|---|---|---|---|---|---|
| easy | 1.00 | +1.35 | 3 | 3 / 3 | Yes |
| medium | 1.00 | +1.75 | 3 | 4 / 4 | Yes |
| hard | — | — | — | — / 8 | TBD |
| **average** | **—** | **—** | **—** | — | **—** |
### WebSocket Protocol

```json
// Reset
{"action": "reset", "task_id": "easy"}

// Step
{"action": {"action_type": "read_data_sample",
            "params": {"table": "raw_orders", "n_rows": 20}}}

// Get state
{"action": "state"}
```

---

## Grading

The grader is fully deterministic:

```
score = assertions_passing / assertions_total   ∈ [0.0, 1.0]
```

Hard task requires both:
- Applying a correct data fix (C1, C2, C3, C4, C6 pass)
- Alerting the correct upstream team for corrupted rows

---

## Project Structure

```
data_pipeline_env/
├── inference.py          ← Baseline LLM agent
├── openenv.yaml          ← OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── src/
    ├── __init__.py
    ├── models.py         ← Pydantic Observation / Action / StepResult
    ├── environment.py    ← DataPipelineEnv (reset / step / state)
    ├── tasks.py          ← Easy / Medium / Hard task definitions
    ├── assertions.py     ← Deterministic assertion checker
    ├── pipeline_runner.py← Pandas-based pipeline executor
    └── server.py         ← FastAPI WebSocket server (OpenEnv spec)
```
