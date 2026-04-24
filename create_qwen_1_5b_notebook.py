"""
Regenerates run_on_kaggle/run_on_kaggle_qwen_1.5b.ipynb from scratch.
Clean text-only pipeline — no Vision-Language remnants.
"""
import json, os

# ── helpers ───────────────────────────────────────────────────────────────
def code_cell(src, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": src.lstrip("\n"),
    }

def md_cell(src, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": src.lstrip("\n"),
    }

# ── cells ─────────────────────────────────────────────────────────────────
CELLS = []

CELLS.append(md_cell("""
# Kaggle Inference — Qwen2.5-1.5B-Instruct (Data Pipeline Incident Response)

Runs the schema-drift agent using **Qwen/Qwen2.5-1.5B-Instruct** (text-only, no vision encoder).

**VRAM budget on Kaggle T4 (15 GB):**
- Model weights in 4-bit NF4: ~0.8 GB
- KV cache + activations (max_new_tokens=1024, history=14 turns): ~1.2 GB
- Framework overhead: ~0.4 GB
- **Total: ~2.4 GB** — leaves 13+ GB headroom, very safe.

---

## Kaggle Secrets required

Add these in **Settings → Secrets** before running:

| Secret name | What it is | Minimum permission |
|---|---|---|
| `GITHUB_TOKEN` | Classic PAT from github.com/settings/tokens | `repo` scope |
| `HF_TOKEN` | HuggingFace token from huggingface.co/settings/tokens | **Read** access |
""", "md-title"))

CELLS.append(code_cell("""
import os, sys
from kaggle_secrets import UserSecretsClient

_s           = UserSecretsClient()
GITHUB_TOKEN = _s.get_secret('GITHUB_TOKEN')
HF_TOKEN     = _s.get_secret('HF_TOKEN')

os.environ['HF_TOKEN']               = HF_TOKEN
os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN

REPO_URL = f'https://{GITHUB_TOKEN}@github.com/standing-on-giants/Meta_hackathon.git'
REPO_DIR = '/kaggle/working/Meta_hackathon'
BRANCH   = 'dev/pratham'

if not os.path.exists(REPO_DIR):
    os.system(f'git clone {REPO_URL} {REPO_DIR}')
else:
    os.system(f'cd {REPO_DIR} && git fetch --all --quiet')

os.system(f'cd {REPO_DIR} && git checkout {BRANCH} || git checkout -b {BRANCH} origin/{BRANCH}')
os.system(f'cd {REPO_DIR} && git pull origin {BRANCH} --quiet')

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# Force-reload src modules so changes from git pull take effect in this session
for _mod in list(sys.modules.keys()):
    if _mod.startswith('src'):
        del sys.modules[_mod]

print('Repo ready:', os.getcwd())
print('Current branch:', BRANCH)
""", "cell-clone"))

CELLS.append(code_cell("""
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers>=4.45.0', 'accelerate', 'bitsandbytes',
    'pandas', 'numpy', 'python-dotenv'], check=True)

import bitsandbytes as bnb, torch
print(f'bitsandbytes : {bnb.__version__}')
print(f'torch        : {torch.__version__}')
print(f'CUDA         : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU          : {torch.cuda.get_device_name(0)}')
    print(f'VRAM         : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
""", "cell-install"))

CELLS.append(code_cell("""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_cfg,
    device_map='auto',
    token=HF_TOKEN,
    torch_dtype=torch.float16,
)
model.eval()

allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved()  / 1e9
total     = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'Model loaded : {MODEL_ID}')
print(f'VRAM alloc   : {allocated:.2f} GB  reserved: {reserved:.2f} GB  free: {total-reserved:.2f} GB')
""", "cell-model"))

CELLS.append(code_cell("""
import re, json, textwrap
from typing import Any, Dict, List, Optional
from src.models import PipelineAction, PipelineObservation

MODEL_NAME  = MODEL_ID
BENCHMARK   = 'data_pipeline_incident_response'

MAX_TOKENS  = int(os.getenv('MAX_TOKENS',   '1024'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))
MAX_STEPS   = int(os.getenv('MAX_STEPS', '100'))

SUCCESS_SCORE_THRESHOLD = 0.1
# Smart fallback logic is now handled dynamically in the runner loop.


# ── OpenEnv stdout logging ─────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f'[START] task={task} env={env} model={model}', flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val   = error if error else 'null'
    action_safe = action.replace('\\n', ' ').replace('\\r', '')
    print(f'[STEP] step={step} action={action_safe} reward={reward:.2f} done={str(done).lower()} error={error_val}', flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f'[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={",".join(f"{r:.2f}" for r in rewards)}', flush=True)


# ── Qwen2.5-1.5B generate (text-only) ─────────────────────────────────────
def _strip_think(text: str) -> str:
    return re.sub(r'<think>[\\s\\S]*?</think>', '', text, flags=re.DOTALL).strip()

def _call_model(messages: list) -> str:
    torch.cuda.empty_cache()
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
    )
    if hasattr(inputs, 'items'):
        inputs = inputs.to(model.device)
        input_len = inputs['input_ids'].shape[1]
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=max(TEMPERATURE, 0.01),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    else:
        inputs = inputs.to(model.device)
        input_len = inputs.shape[1]
        with torch.no_grad():
            out_ids = model.generate(
                inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=max(TEMPERATURE, 0.01),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    raw = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)
    return _strip_think(raw)


# ── Action parser ──────────────────────────────────────────────────────────
def parse_llm_response(text: str) -> Optional[PipelineAction]:
    if not text:
        return None
    text = text.strip()
    if '```' in text:
        text = '\\n'.join(l for l in text.split('\\n') if not l.strip().startswith('```'))
    start = text.find('{')
    if start == -1:
        return None
    end = text.rfind('}') + 1
    if end > start:
        try:
            data = json.loads(text[start:end])
            if isinstance(data, dict) and 'action_type' in data:
                return PipelineAction(**{'action_type': data['action_type'], 'params': data.get('params', {})})
        except Exception:
            pass
    # Repair truncated JSON
    for suffix in ['\"}}', '}}', '}']:
        try:
            data = json.loads(text[start:] + suffix)
            if isinstance(data, dict) and 'action_type' in data:
                return PipelineAction(**{'action_type': data['action_type'], 'params': data.get('params', {})})
        except Exception:
            continue
    return None


print(f'Config ready. MAX_STEPS={MAX_STEPS}, MAX_TOKENS={MAX_TOKENS}, TEMPERATURE={TEMPERATURE}')

# ── Context safety limits for 1.5B model ──────────────────────────────────
# 1.5B model has a 32k context, but long histories cause silent OOM / empty output.
# Keep the prompt short: cap history and truncate the user message.
MAX_HISTORY_TURNS  = 6      # max (user, assistant) pairs kept in history
MAX_PROMPT_CHARS   = 3000   # truncate user prompt beyond this many chars
""", "cell-config"))

CELLS.append(code_cell("""
SYSTEM_PROMPT = textwrap.dedent(\"\"\"
You are an expert data engineer diagnosing and fixing broken data pipelines.
Choose ONE action each turn. Respond with ONLY a single JSON object. No markdown, no prose.

WORKFLOW:
1. read_data_sample on the raw input table first.
2. check_schema / compare_schema on the raw input table if drift is suspected.
3. If compare_schema shows renamed/missing columns, call handle_drift first.
4. Apply fix: add_data_filter or patch_transformation.
5. run_pipeline to verify. Repeat if still failing.
6. alert_upstream_team ONLY when data is genuinely unfixable.

AVAILABLE ACTIONS (respond with ONLY a JSON object):
{"action_type": "read_data_sample", "params": {"table": "raw_input_table", "n_rows": 20}}
{"action_type": "check_schema", "params": {"table": "raw_input_table"}}
{"action_type": "compare_schema", "params": {"table": "raw_input_table"}}
{"action_type": "handle_drift", "params": {"strategy": "resolve_column_rename", "table": "raw_input_table", "old_column": "old", "new_column": "new"}}
{"action_type": "add_data_filter", "params": {"step_id": "step_id", "filter_condition": "col IS NOT NULL"}}
{"action_type": "patch_transformation", "params": {"step_id": "step_id", "patch_type": "coalesce|cast_column|dedup|parse_currency|strip_prefix", "column": "col_name"}}
{"action_type": "alert_upstream_team", "params": {"team": "vendor_support", "issue_description": "corrupted"}}
{"action_type": "run_pipeline", "params": {}}

PATCH TYPES: cast_column | coalesce | dedup | parse_currency | strip_prefix
IMPORTANT: After parse_currency, ALWAYS chain coalesce on the same column before run_pipeline.
Exception: if column is a denominator (e.g. impressions in CTR), filter IS NOT NULL instead of coalesce.

HANDLE DRIFT strategies: detect | resolve_column_rename | numeric_format | null_fill |
  type_cast | join_key_prefix | filter_invalid | alert_upstream

FILTER OPERATORS (only these are supported): IS NOT NULL | IS NULL | >= | <=

RULES:
- ONLY JSON. No explanation.
- Never fix before reading data (-0.5 penalty).
- You can repeat actions, but repeating the same check or re-running the pipeline without applying a new fix wastes your limited step budget and incurs negative penalties. Only repeat actions if it makes sense to gather new information or verify a fix.
- Stop when pipeline_passed is true.
\"\"\").strip()


def _detect_action_loop(actions_taken: List[str]) -> Optional[str]:
    if len(actions_taken) < 3:
        return None
    def _key(s):
        idx = s.find(']')
        return s[idx+1:].strip() if idx >= 0 else s.strip()
    last = [_key(a) for a in actions_taken[-3:]]
    if last[0] == last[1] == last[2]:
        return last[0]
    return None


def build_prompt(obs, step: int) -> str:
    failed  = '\\n'.join(
        f'  [{r.assertion_id}] {r.assertion_type} on {r.table}({r.column or "N/A"}): {r.actual}'
        for r in obs.failed_assertions
    ) or '  (none)'
    passed  = ', '.join(r.assertion_id for r in obs.passed_assertions) or 'none'
    dag     = '\\n'.join(
        f'  {n.step_id}: {n.input_table} -> {n.output_table}'
        + (f' | filters:{n.applied_filters}' if n.applied_filters else '')
        + (f' | patches:{n.applied_patches}' if n.applied_patches else '')
        for n in obs.dag_structure
    )
    hist    = '\\n'.join(f'  {r.date}: {r.status} ({r.row_count} rows)' for r in obs.historical_runs[-2:])
    actions = '\\n'.join(f'  {a}' for a in obs.actions_taken[-20:]) or '  (none)'
    sample = ''
    if obs.data_sample:
        rows      = obs.data_sample[:4]
        null_rows = [r for r in obs.data_sample if any(v is None for v in r.values())]
        sample    = '\\nDATA SAMPLE:\\n' + json.dumps(rows, default=str)
        if null_rows:
            sample += f'\\nNULL ROWS ({len(null_rows)} found):\\n' + json.dumps(null_rows[:3], default=str)
    schema = ''
    if obs.current_schema:
        schema += '\\nSCHEMA: ' + json.dumps(obs.current_schema)
    if obs.schema_diff:
        schema += '\\nSCHEMA DIFF: ' + json.dumps(obs.schema_diff)
    loop = _detect_action_loop(obs.actions_taken)
    hint = f'\\n[LOOP DETECTED] Stop repeating {loop}. Try a completely different action.' if loop else ''
    steps_remaining = MAX_STEPS - step
    return textwrap.dedent(f\"\"\"
    STEP {step}/{MAX_STEPS}  |  REMAINING: {steps_remaining}
    TASK: {obs.task_id} ({obs.difficulty})
    DESCRIPTION: {obs.description}
    PIPELINE PASSED: {obs.pipeline_passed}
    LAST ACTION RESULT: {obs.last_action_result}
    DAG:
    {dag}
    FAILING:
    {failed}
    PASSING: {passed}
    HISTORY:
    {hist}
    RECENT ACTIONS:
    {actions}
    {sample}{schema}{hint}
    Respond with exactly ONE action JSON object.
    \"\"\").strip()


print('Prompts ready.')
""", "cell-prompts"))

CELLS.append(code_cell("""
from src.environment import DataPipelineEnv


def run_episode(task_id: str, max_steps: int = MAX_STEPS, verbose: bool = True) -> Dict[str, Any]:
    # Defensive creation: supports both old env (no max_steps kwarg) and new env
    try:
        env = DataPipelineEnv(task_id=task_id, max_steps=max_steps)
    except TypeError:
        env = DataPipelineEnv(task_id=task_id)
        env.MAX_STEPS = max_steps

    history:         List[Dict[str, str]] = []
    rewards:         List[float]          = []
    steps_taken:     int                  = 0
    score:           float                = 0.0
    success:         bool                 = False
    n_passed:        int                  = 0
    n_total:         int                  = 0
    pipeline_passed: bool                 = False
    consecutive_errors: int               = 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        if verbose:
            print(f'\\n{"="*60}', file=sys.stderr)
            print(f'TASK: {task_id.upper()}  |  {len(obs.failed_assertions)} assertions failing', file=sys.stderr)
            print(f'{"="*60}', file=sys.stderr)

        for step in range(1, max_steps + 1):
            if obs.pipeline_passed:
                if verbose:
                    print(f'\\n[PASSED] Pipeline passed at step {step - 1}!', file=sys.stderr)
                break

            # Loop detection: trim history to break repetitive context
            if _detect_action_loop(obs.actions_taken):
                history = history[-2:]
                consecutive_errors = 0

            # After 3 consecutive generation errors, nuke history and try fresh
            if consecutive_errors >= 3:
                print(f'[RESET] {consecutive_errors} consecutive errors. Clearing history.', file=sys.stderr)
                history = []
                torch.cuda.empty_cache()
                consecutive_errors = 0

            user_prompt = build_prompt(obs, step)
            # Hard cap on prompt length to prevent context overflow on 1.5B
            if len(user_prompt) > MAX_PROMPT_CHARS:
                user_prompt = user_prompt[:MAX_PROMPT_CHARS] + '\\n[TRUNCATED]\\nRespond with exactly ONE action JSON object.'
            history.append({'role': 'user', 'content': user_prompt})

            # Aggressively cap history for 1.5B model
            if len(history) > MAX_HISTORY_TURNS * 2:
                history = history[-(MAX_HISTORY_TURNS * 2):]

            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + history

            response_text = ''
            try:
                response_text = _call_model(messages)
                consecutive_errors = 0
            except torch.cuda.OutOfMemoryError:
                consecutive_errors += 1
                print(f'[OOM] Step {step}: trimming to 2 turns and retrying.', file=sys.stderr)
                history = history[-2:]
                messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + history
                torch.cuda.empty_cache()
                try:
                    response_text = _call_model(messages)
                    consecutive_errors = 0
                except Exception as e2:
                    print(f'[OOM-RETRY-FAIL] {type(e2).__name__}: {e2}', file=sys.stderr)
            except Exception as exc:
                consecutive_errors += 1
                if verbose:
                    print(f'  [Step {step}] Generation error ({type(exc).__name__}): {exc}', file=sys.stderr)
                # Trim history on any error to reduce context pressure next step
                history = history[-4:]

            action = parse_llm_response(response_text)

            if action is None:
                target_table = None
                if obs.failed_assertions:
                    target_table = obs.failed_assertions[0].table
                elif obs.dag_structure:
                    target_table = obs.dag_structure[0].input_table
                else:
                    target_table = "unknown_table"
                action = PipelineAction(
                    action_type='read_data_sample',
                    params={'table': target_table, 'n_rows': 20}
                )

            history.append({'role': 'assistant', 'content': response_text or '{}'})

            result = env.step(action)
            obs    = result.observation
            reward = result.reward or 0.0
            done   = result.done
            error  = getattr(obs, 'last_action_error', None) or None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=json.dumps(action.model_dump()), reward=reward, done=done, error=error)

            if verbose:
                print(f'[Step {step}] {action.action_type}({action.params})  reward={reward:+.2f}  '
                      f'passed={len(obs.passed_assertions)}/{len(obs.failed_assertions)+len(obs.passed_assertions)}  '
                      f'| {obs.last_action_result[:80]}', file=sys.stderr)

            if done:
                break

        n_total         = len(obs.failed_assertions) + len(obs.passed_assertions)
        n_passed        = len(obs.passed_assertions)
        pipeline_passed = obs.pipeline_passed
        score           = min(max(n_passed / n_total if n_total > 0 else 0.0, 0.01), 0.99)
        success         = score >= SUCCESS_SCORE_THRESHOLD

        if verbose:
            print(f'\\n--- Episode Summary ---', file=sys.stderr)
            print(f'  Score: {score:.2f}  Reward: {sum(rewards):.2f}  Steps: {steps_taken}/{max_steps}  Passed: {pipeline_passed}', file=sys.stderr)

    except Exception as exc:
        import traceback
        print(f'[ERROR] {task_id}: {exc}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        'task_id':           task_id,
        'score':             round(score, 4),
        'success':           success,
        'pipeline_passed':   pipeline_passed,
        'total_reward':      round(sum(rewards), 4),
        'steps_taken':       steps_taken,
        'assertions_passed': n_passed,
        'assertions_total':  n_total,
    }


print('Runner ready.')
""", "cell-runner"))

CELLS.append(code_cell("""
from src.tasks import TASKS as _AVAILABLE_TASKS

ALL_TASKS   = ['easy', 'medium', 'hard', 'hard2']
VALID_TASKS = [t for t in ALL_TASKS if t in _AVAILABLE_TASKS]
print(f'Available tasks: {VALID_TASKS}', file=sys.stderr)

# Change to run a single task: 'easy' | 'medium' | 'hard' | 'hard2' | 'all'
TASKS_TO_RUN = 'all'

task_ids = VALID_TASKS if TASKS_TO_RUN == 'all' else (
    [TASKS_TO_RUN] if TASKS_TO_RUN in _AVAILABLE_TASKS else []
)

all_results = []
for task_id in task_ids:
    result = run_episode(task_id=task_id, max_steps=MAX_STEPS, verbose=True)
    all_results.append(result)
    torch.cuda.empty_cache()

print('\\n' + '='*60, file=sys.stderr)
print('FINAL SCORES', file=sys.stderr)
print('='*60, file=sys.stderr)
total_score = 0.0
for r in all_results:
    status = '[PASSED]' if r['pipeline_passed'] else '[FAILED]'
    print(f\"  {r['task_id']:8s} | score={r['score']:.2f} | reward={r['total_reward']:+.2f} | steps={r['steps_taken']:2d} | {status}\", file=sys.stderr)
    total_score += r['score']

avg = total_score / len(all_results) if all_results else 0.0
print(f'\\n  Avg score: {avg:.4f}', file=sys.stderr)
print('\\nJSON_RESULTS:', json.dumps(all_results, indent=2), file=sys.stderr)
""", "cell-run"))

CELLS.append(code_cell("""
# Run any time to check VRAM health
allocated = torch.cuda.memory_allocated() / 1e9
reserved  = torch.cuda.memory_reserved()  / 1e9
total_vr  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'VRAM allocated : {allocated:.2f} GB')
print(f'VRAM reserved  : {reserved:.2f} GB')
print(f'VRAM free      : {total_vr - reserved:.2f} GB  (of {total_vr:.1f} GB)')
""", "cell-vram"))

# ── assemble notebook ──────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True,
        },
    },
    "cells": CELLS,
}

out = "run_on_kaggle/run_on_kaggle_qwen_1.5b.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Written: {out}  ({os.path.getsize(out)//1024} KB)")
