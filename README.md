---
title: Legal Contract Review — OpenEnv
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
  - legal
  - rl-environment
  - agentic
license: mit
---

# Legal Contract Review — OpenEnv

An RL environment where an agent reviews legal contracts, identifies risks, detects missing clauses, and proposes fixes.

---

## Motivation

Legal teams review contracts daily — NDAs, SaaS agreements, M&A term sheets. Missing a liability cap or a buried auto-renewal clause can cost organisations millions. This environment simulates that workflow: the agent acts as a junior associate, reading sections sequentially, flagging risky or non-standard clauses, detecting absent protections, suggesting replacement language, and producing a final signed-off review.

The environment benchmarks agentic LLMs on tasks that require **sequential reasoning under a step budget**, **precision** (false positives are penalised), and **discrimination** between genuinely risky clauses and market-standard ones.

---

## Project Structure

```
legal_contract_env/
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── inference.py          Agent loop, prompt builder, action parser (local use)
└── src/
    ├── __init__.py
    ├── contracts.py      Synthetic contracts + ground-truth fault manifests
    ├── environment.py    OpenEnv-compliant env (reset / step / state)
    ├── grader.py         Deterministic grader (F1-weighted, recall-focused)
    ├── models.py         Pydantic models (ContractAction, ContractObservation, …)
    └── server.py         FastAPI REST wrapper
```

---

## Setup

### Local (Docker)

```bash
docker build -t legal-contract-env .
docker run -p 7860:7860 legal-contract-env
```

### Local (Python)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn src.server:app --host 0.0.0.0 --port 7860
```

### Hugging Face Space

Push this repo to a Hugging Face Space with `sdk: docker` in the README frontmatter (already set above). The Space builds and serves automatically.

```bash
git clone https://huggingface.co/spaces/<your-org>/legal-contract-review
cp -r . legal-contract-review/
cd legal-contract-review
git add . && git commit -m "initial" && git push
```

---

## Environment variables (local inference only)

These are only needed when running `inference.py` locally. The Space itself has no model dependency — it only serves the environment API.

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes (local) | — | API key for Groq inference |
| `MODEL_NAME` | No | `llama-3.3-70b-versatile` | Model override |
| `MAX_STEPS` | No | `30` | Max steps per episode |
| `TEMPERATURE` | No | `0.1` | Sampling temperature |
| `MAX_TOKENS` | No | `500` | Max tokens per LLM call |

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
  -d '{"task_id": "easy", "max_steps": 30}'
# → {"session_id": "easy_...", "observation": {...}}

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "<session_id from reset>",
    "action": {
      "action_type": "read_section",
      "params": {"section": "obligations"}
    }
  }'

# Get session state
curl "http://localhost:7860/state?session_id=<session_id>"
```

### Local agent runner

```bash
pip install ollama   # in addition to requirements.txt
export GROQ_API_KEY="your_key_here"

python inference.py              # all three tasks
python inference.py --task easy
python inference.py --task hard --steps 35
python inference.py --quiet      # suppress per-step output
```

---

## Environment Description

```python
from src.environment import LegalContractEnv
from src.models import ContractAction

env = LegalContractEnv(task_id="easy")  # "easy" | "medium" | "hard"
obs = env.reset()                        # → ContractObservation
result = env.step(action)                # → StepResult
state = env.state()                      # → Dict
```

Episode ends when the agent calls `summarize` or the step budget is exhausted. `pipeline_passed = True` when the grader score ≥ 0.6.

---

## Observation Space

`ContractObservation` fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | `"easy"`, `"medium"`, or `"hard"` |
| `difficulty` | `str` | Human-readable difficulty label |
| `description` | `str` | Task description |
| `max_steps` | `int` | Step budget |
| `contract_title` | `str` | Title of the contract under review |
| `available_sections` | `List[str]` | All section names |
| `section_statuses` | `List[SectionStatus]` | Per-section read / approved / flag state |
| `current_section_text` | `Optional[str]` | Full text of the last read section |
| `current_section_name` | `Optional[str]` | Name of the last read section |
| `flags` | `List[AgentFlag]` | All flags raised so far |
| `actions_taken` | `List[str]` | Recent action history (last 8) |
| `last_action_result` | `str` | Result of the last action |
| `step` | `int` | Current step number |
| `done` | `bool` | Whether the episode has ended |
| `pipeline_passed` | `bool` | `True` when score ≥ 0.6 after `summarize` |
| `total_faults_in_contract` | `int` | Ground-truth fault count (traps excluded) |
| `faults_found_so_far` | `int` | Real faults matched so far |

`SectionStatus`: `section_name`, `read`, `approved`, `flags_count`

`AgentFlag`: `section`, `clause_id`, `flag_type` (`"risky"` / `"missing"`), `risk_level`, `reason`, `redline_suggested`

---

## Action Space

| Action | Params | Reward |
|---|---|---|
| `read_section` | `section` | +0.05 |
| `flag_clause` | `section`, `clause_id`, `risk_level`, `reason` | +0.10 |
| `mark_missing` | `section`, `clause_id`, `risk_level`, `reason` | +0.10 |
| `suggest_redline` | `clause_id`, `replacement_text` | +0.05 |
| `approve_section` | `section` | +0.02 |
| `summarize` | *(none)* | grader reward |

Penalty of −0.20 if `flag_clause` or `mark_missing` is called on a section that has not been read first.

---

## Grading

Score formula (computed at `summarize`):

```
recall    = true_positives / total_real_faults
precision = true_positives / (true_positives + false_positives)
f_score   = 2 × recall × precision / (recall + precision)
score     = max(0, f_score − 0.3 × missed_criticals / total_faults)
```

Grader rewards at `summarize`:

| Event | Reward |
|---|---|
| True positive — critical | +0.80 |
| True positive — medium | +0.60 |
| True positive — low | +0.40 |
| Correct `risk_level` | +0.10 |
| Redline matches standard language | +0.30 |
| Missed critical fault | −1.00 |

---

## Task Descriptions

### easy — Mutual NDA

Sections: `parties`, `purpose`, `definition_confidential`, `obligations`, `term`, `governing_law`, `general`

| Fault | Type | Section | Risk |
|---|---|---|---|
| Missing liability cap | Missing clause | `obligations` | Critical |
| Uncapped one-sided indemnification | Risky clause | `obligations` | Critical |

**Difficulty:** Both faults are in the same section and readable on a single pass. Expected to be caught within 15 steps.

---

### medium — SaaS Subscription Agreement

Sections: `definitions`, `license_grant`, `fees_payment`, `data_privacy`, `intellectual_property`, `warranties`, `limitation_liability`, `term_termination`, `general`

| Fault | Type | Section | Risk |
|---|---|---|---|
| Auto-renewal + 15% price escalation buried in Definitions | Risky clause | `definitions` | Medium |
| Irrevocable perpetual sublicensable data license surviving termination | Risky clause | `intellectual_property` | Critical |
| No SLA or uptime commitment | Missing clause | `data_privacy` | Medium |

**Difficulty:** Moderate. Requires recognising a predatory drafting pattern and detecting a missing clause by absence.

---

### hard — M&A Term Sheet

Sections: `transaction_summary`, `purchase_price_adjustment`, `representations_warranties`, `indemnification`, `intellectual_property`, `employee_matters`, `conditions_closing`, `exclusivity_no_shop`, `schedule_a_open_source`, `schedule_b_earnout_definition`

| Fault | Type | Section | Risk |
|---|---|---|---|
| GPLv3 copyleft: 34% of codebase, triggers on distribution | Risky clause | `schedule_a_open_source` | Critical |
| Earnout ARR in Acquirer's sole discretion + CFO gate on channel revenue | Risky clause | `schedule_b_earnout_definition` | Medium |
| No R&W insurance | Missing clause | `conditions_closing` | Medium |
| 1% tipping basket *(trap — market standard, do not flag)* | — | `indemnification` | — |
| 18-month rep survival *(trap — market standard, do not flag)* | — | `representations_warranties` | — |

**Difficulty:** Hard. The agent must read schedules, understand GPLv3 copyleft mechanics, and correctly avoid flagging the two trap clauses.

---

## Baseline Scores

Model: `glm-5:cloud` via Ollama. `MAX_STEPS=30`, `TEMPERATURE=0.1`.

| Task | Score | Reward | Steps | Faults caught | Passed |
|---|---|---|---|---|---|
| easy | 1.00 | +2.82 | 15 | 2 / 2 | Yes |
| medium | 1.00 | +3.60 | 25 | 3 / 3 | Yes |
| hard | 1.00 | +4.19 | 24 | 3 / 3 | Yes |
| **average** | **1.00** | **+3.54** | **21** | — | **3 / 3** |
