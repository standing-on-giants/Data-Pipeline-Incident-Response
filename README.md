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

---

## Setup

### Local (Docker)

```bash
docker build -t data-pipeline-env .
docker run -p 7860:7860 data-pipeline-env
```

### Local (Python conda/venv)

```bash
pip install -r requirements.txt
uvicorn src.server:app --host 0.0.0.0 --port 7860
```

### Hugging Face Space

Push this repo to a Hugging Face Space with `sdk: docker` in the README frontmatter. The Space builds and serves automatically.

```bash
git clone https://huggingface.co/spaces/<your-org>/data-pipeline-incident-response
cp -r . data-pipeline-incident-response/
cd data-pipeline-incident-response
git add . && git commit -m "initial" && git push
```

---

## Environment variables (local inference only)

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

---

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
