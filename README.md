# Data Pipeline Incident Response — OpenEnv

An RL environment where an agent diagnoses and fixes broken data pipelines.

**Real-world task**: Every data team deals with pipeline failures overnight. The agent
acts as an on-call data engineer: it reads failing assertion reports, inspects raw data,
identifies root causes, and applies fixes — exactly what a human would do at 3am.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run inference against all 3 tasks (requires API key)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_token_here"

python inference.py
python inference.py --task easy    # single task
python inference.py --task hard --steps 25
```

---

## Environment Overview

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