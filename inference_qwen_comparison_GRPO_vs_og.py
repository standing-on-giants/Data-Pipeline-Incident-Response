import os, sys, json, textwrap, re, torch, argparse
from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.environment import DataPipelineEnv
from src.models import PipelineAction, PipelineObservation
from src.tasks import TASKS as _AVAILABLE_TASKS

BASE_MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'
SFT_DIR = '/kaggle/working/sft_qwen'
GRPO_DIR = '/kaggle/working/grpo_qwen'
LOCAL_MERGED_DIR = '/kaggle/working/qwen-merged-16bit'
HF_REPO = 'Abhinav-hf/data-pipeline-incident-qwen-grpo'

try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    HF_TOKEN = _s.get_secret('HF_TOKEN')
except Exception:
    HF_TOKEN = os.getenv('HF_TOKEN')

MAX_TOKENS = 512
TEMPERATURE = 0.1
MAX_STEPS = 25
BENCHMARK = 'data_pipeline_incident_response'
SUCCESS_SCORE_THRESHOLD = 0.1
FALLBACK_ACTION = PipelineAction(action_type='compare_schema', params={'table': 'insights_ads'})

def _strip_think(text: str) -> str:
    return re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL).strip()

def generate(model, tokenizer, messages: list) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors='pt', padding=False).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=max(TEMPERATURE, 0.01),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs['input_ids'].shape[1]
    raw = tokenizer.decode(out_ids[0][input_len:], skip_special_tokens=True)
    return _strip_think(raw)

# ------------------------------------------------------------------ #
# Action parser
# ------------------------------------------------------------------ #

def parse_llm_response(text: str) -> PipelineAction:
    payload = _extract_action_payload(text)
    if not payload:
        return FALLBACK_ACTION
    normalized = _normalize_action_payload(payload)
    try:
        return PipelineAction(**normalized)
    except Exception:
        return FALLBACK_ACTION

def _extract_action_payload(text: str) -> Optional[dict]:
    if not text: return None
    text = text.strip()
    if '```' in text:
        lines = text.split('\n')
        text = '\n'.join(l for l in lines if not l.strip().startswith('```'))
    start = text.find('{')
    if start == -1: return None
    end = text.rfind('}') + 1
    if end > start:
        try:
            data = json.loads(text[start:end])
            if isinstance(data, dict) and 'action_type' in data: return data
        except Exception: pass
    fragment = text[start:]
    repaired = _try_repair_json(fragment)
    if repaired and isinstance(repaired, dict): return repaired
    return None

def _normalize_action_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    action_type = str(payload.get('action_type', '')).strip()
    params = payload.get('params') or {}
    if not isinstance(params, dict): params = {}
    if action_type != 'handle_drift':
        return {'action_type': action_type, 'params': params}
    return {'action_type': 'handle_drift', 'params': params}

def _try_repair_json(fragment: str) -> Optional[dict]:
    for suffix in ['"}}}', '"}}}', '"}}', '}}', '}']:
        try:
            data = json.loads(fragment + suffix)
            if isinstance(data, dict) and 'action_type' in data: return data
        except Exception: continue
    at_match = re.search(r'"\"action_type\"\\s*:\\s*\"([^\"]+)\"', fragment)
    if not at_match: at_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', fragment)
    if not at_match: return None
    action_type = at_match.group(1)
    params: Dict[str, Any] = {}
    step_match   = re.search(r'"step_id"\s*:\s*"([^"]+)"', fragment)
    patch_match  = re.search(r'"patch_type"\s*:\s*"([^"]+)"', fragment)
    col_match    = re.search(r'"column"\s*:\s*"([^"]+)"', fragment)
    table_match  = re.search(r'"table"\s*:\s*"([^"]+)"', fragment)
    filter_match = re.search(r'"filter_condition"\s*:\s*"([^"]+)"', fragment)
    if step_match:   params['step_id']           = step_match.group(1)
    if patch_match:  params['patch_type']         = patch_match.group(1)
    if col_match:    params['column']             = col_match.group(1)
    if table_match:  params['table']              = table_match.group(1)
    if filter_match: params['filter_condition']   = filter_match.group(1)
    try:
        return {'action_type': action_type, 'params': params}
    except Exception: return None

# ------------------------------------------------------------------ #
# Prompts
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data engineer diagnosing and fixing broken data pipelines.

You will receive the current state of a pipeline (failing assertions, DAG structure,
historical run info) and must choose ONE action to take each turn.

WORKFLOW (follow this order strictly):
1. FIRST: read_data_sample on the raw input table(s) to see what the data looks like.
2. THEN: Use check_schema or compare_schema if a type or schema issue is suspected.
3. If you see any schema drift signal (renamed/missing columns, changed types, auth format drift,
   or stricter rate-limit behavior), use handle_drift.
4. THEN: Apply the RIGHT fix using add_data_filter or patch_transformation.
5. THEN: Call run_pipeline to verify the fix.
6. ONLY AFTER fixing what you can: If some data is genuinely corrupted (e.g. "N/A" values
   that cannot be parsed), call alert_upstream_team.
7. If assertions are still failing after run_pipeline, investigate more and apply
   additional fixes. Do NOT just call run_pipeline again without changing something.

AVAILABLE ACTIONS (respond with ONLY a JSON object, no markdown, no prose):

{"action_type": "read_data_sample", "params": {"table": "<table_name>", "n_rows": 20}}
{"action_type": "check_schema", "params": {"table": "<table_name>"}}
{"action_type": "compare_schema", "params": {"table": "<table_name>"}}
{"action_type": "handle_drift", "params": {"strategy": "<detect|numeric_format|null_fill|type_cast|join_key_prefix|filter_invalid|resolve_column_rename|alert_upstream>", "table": "<table_name_optional>", "step_id": "<step_id_optional>", "column": "<column_optional>", "old_column": "<optional_old_name>", "new_column": "<optional_new_name>", "filter_condition": "<optional>", "team": "<optional>", "issue_description": "<optional>"}}
{"action_type": "run_quality_assertion", "params": {"assertion_id": "<e.g. A1>"}}
{"action_type": "add_data_filter", "params": {"step_id": "<step_id>", "filter_condition": "<e.g. user_id IS NOT NULL>"}}
{"action_type": "patch_transformation", "params": {"step_id": "<step_id>", "patch_type": "<cast_column|coalesce|dedup|parse_currency|strip_prefix>", "column": "<column_name>"}}
{"action_type": "backfill_partition", "params": {"date": "<YYYY-MM-DD>"}}
{"action_type": "alert_upstream_team", "params": {"team": "<team_name_snake_case>", "issue_description": "<short description>"}}
{"action_type": "mark_acceptable", "params": {"assertion_id": "<id>", "reason": "<reason>"}}
{"action_type": "run_pipeline", "params": {}}

KEY PATCH TYPES (you can chain multiple patches on the same step — they run in order):
- parse_currency: Use when a column has mixed formats like "$1,234.56" and "1234.56" and "N/A".
  It strips $, commas, casts to float, and converts N/A to NaN. Works on ANY column with "N/A" strings,
  not just currency — e.g. if a numeric column like impressions has "N/A" values, use parse_currency on it.
- coalesce: Use AFTER parse_currency to replace NaN/null with a default value (default is 0).
  IMPORTANT: After parse_currency, NaN values will cause value_range assertions to fail.
  You MUST chain a coalesce patch to fix this: {"action_type": "patch_transformation", "params": {"step_id": "<same_step>", "patch_type": "coalesce", "column": "<same_column>"}}
  If coalescing a denominator column (e.g. impressions used in CTR = clicks/impressions), coalescing to 0
  will cause division by zero. Instead, filter out those rows: add_data_filter with "column IS NOT NULL".
- cast_column: Use when a column needs simple numeric casting.
- dedup: Use when there are duplicate rows based on a key column.
  IMPORTANT: If a "unique" assertion is failing, the fix is ALWAYS dedup on the failing column.
  Do NOT use coalesce or add_data_filter for uniqueness failures — only dedup works.
- strip_prefix: Use when column values have a spurious prefix like "CMP_" that needs removal.
  Params: step_id, column. Optionally "prefix" (default "CMP_"). After stripping, chain cast_column
  if the underlying value should be numeric.

DRIFT HANDLING RULES:
- Use handle_drift when schema or contract changes between runs.
- handle_drift strategy mapping:
    detect -> compare_schema
    numeric_format -> patch_transformation(parse_currency)
    null_fill -> patch_transformation(coalesce)
    type_cast -> patch_transformation(cast_column)
    join_key_prefix -> patch_transformation(strip_prefix)
    filter_invalid -> add_data_filter
    resolve_column_rename -> restore compatibility for renamed columns (e.g. spend <- total_spend)
    alert_upstream -> alert_upstream_team
- For spend -> total_spend style drift, compare schema first, then patch transformations to align types.

UPSTREAM TEAM NAMING:
- Team names are always lowercase snake_case. Examples: meta_ads_api_team, data_engineering, vendor_support.
- If the description mentions "Meta", "Graph API", or "Meta Ads", the team is likely "meta_ads_api_team".

RULES:
- RESPOND WITH ONLY A JSON OBJECT. No markdown fences, no explanation, no prose.
- Do NOT call run_pipeline unless you applied a filter or patch since the last run.
- Do NOT apply a fix before reading the data — this will be penalised.
- NEVER use mark_acceptable. It always results in a heavy penalty. Instead, fix the data.
- After parse_currency, ALWAYS chain coalesce on the same column to handle NaN values before calling run_pipeline.
- If a "unique" assertion fails (e.g. uniqueness on order_item_id), the ONLY correct fix is dedup.
  Do NOT try coalesce, add_data_filter, or any other patch for uniqueness failures.
- If a computed column (like CTR) has a value_range failure, check ALL input columns in its formula.
  For example, if CTR = clicks/impressions and impressions has "N/A" strings, you must fix impressions
  with parse_currency first, then filter out null rows, before the computed column can produce valid values.
- If a joined output table has 0 rows (row_count assertion failing), the join keys likely don't match.
  Use compare_schema on the input tables to detect type/format drifts like string vs int, or unwanted
  prefixes on the join key. Apply strip_prefix + cast_column to align the keys.
- If pipeline_passed is true, you are done — unless the task description mentions alerting an upstream team.
- NEVER repeat the same action you already tried. If an action did not fix the problem, try a DIFFERENT action.
""").strip()

def _collect_schema_drift_signals(obs: PipelineObservation) -> List[str]:
    signals = []
    desc = (obs.description or "").lower()
    if "schema drift" in desc or "contract" in desc: signals.append("Task description references schema/contract drift.")
    if obs.schema_diff:
        schema_diff_json = json.dumps(obs.schema_diff).lower()
        if "removed" in schema_diff_json: signals.append("Historical columns appear removed in current schema.")
        if "changed" in schema_diff_json: signals.append("Column types differ from historical schema.")
        if "new" in schema_diff_json: signals.append("New columns detected relative to historical schema.")
    for r in obs.failed_assertions:
        actual = (r.actual or "").lower()
        if "missing" in actual and "column" in actual: signals.append(f"Assertion {r.assertion_id} reports a missing column.")
        if "not found" in actual and "column" in actual: signals.append(f"Assertion {r.assertion_id} reports a renamed or deleted column.")
        if "type" in actual and ("object" in actual or "string" in actual): signals.append(f"Assertion {r.assertion_id} indicates possible type drift.")
    deduped = []
    for s in signals:
        if s not in deduped: deduped.append(s)
    return deduped[:6]

def _detect_action_loop(actions_taken: List[str]) -> Optional[str]:
    if len(actions_taken) < 3: return None
    def _extract_action_key(action_str: str) -> str:
        idx = action_str.find(']')
        if idx >= 0: return action_str[idx+1:].strip()
        return action_str.strip()
    last_3 = [_extract_action_key(a) for a in actions_taken[-3:]]
    if last_3[0] == last_3[1] == last_3[2]: return last_3[0]
    if len(actions_taken) >= 4:
        last_4 = [_extract_action_key(a) for a in actions_taken[-4:]]
        if last_4[0] == last_4[2] and last_4[1] == last_4[3]:
            return f"{last_4[0]} and {last_4[1]} (oscillating)"
    return None

def _build_loop_hint(obs: PipelineObservation, looped_action: str) -> str:
    hint = f"\\n[CRITICAL LOOP DETECTED]: You have been repeating '{looped_action}' without making progress. You MUST try a COMPLETELY DIFFERENT action type.\\n"
    for r in obs.failed_assertions:
        atype = (r.assertion_type or "").lower()
        col = r.column or ""
        table = r.table or ""
        if atype == "unique": hint += f"\\n  -> Assertion {r.assertion_id} uniquely failed on '{col}'. Fix: dedup."
        elif atype == "row_count": hint += f"\\n  -> Assertion {r.assertion_id} row count failed on '{table}'. Fix dedup first."
        elif atype == "type_check": hint += f"\\n  -> Assertion {r.assertion_id} type check failed on '{col}'. Fix: parse_currency."
        elif atype == "value_range": hint += f"\\n  -> Assertion {r.assertion_id} value range failed on '{col}'. Fix: coalesce."
    if "run_pipeline" in looped_action:
        hint += "\\n\\n  You MUST apply a fix BEFORE calling run_pipeline again."
    return hint

def build_user_prompt(obs: PipelineObservation, step: int) -> str:
    failed_str = "\\n".join(f"  [{r.assertion_id}] {r.assertion_type} on {r.table}({r.column or 'N/A'}): {r.actual}" for r in obs.failed_assertions) or "  (none -- all passing!)"
    passed_str = ", ".join(r.assertion_id for r in obs.passed_assertions) or "none"
    dag_str = "\\n".join(f"  {n.step_id}: {n.input_table} -> {n.output_table}" + (f" | filters: {n.applied_filters}" if n.applied_filters else "") + (f" | patches: {n.applied_patches}" if n.applied_patches else "") for n in obs.dag_structure)
    hist_str = "\\n".join(f"  {r.date}: {r.status} ({r.row_count} rows)" for r in obs.historical_runs)
    sample_str = ""
    if obs.data_sample:
        sample_rows = obs.data_sample[:5]
        null_rows = [r for r in obs.data_sample if any(v is None for v in r.values())]
        if null_rows:
            sample_str = "\\nDATA SAMPLE (first 5 rows):\\n" + json.dumps(sample_rows, indent=2, default=str) + f"\\nROWS WITH NULL VALUES:\\n" + json.dumps(null_rows[:3], indent=2, default=str)
        else:
            sample_str = "\\nDATA SAMPLE:\\n" + json.dumps(sample_rows, indent=2, default=str)
    schema_str = ""
    if obs.current_schema: schema_str = "\\nCURRENT SCHEMA: " + json.dumps(obs.current_schema)
    if obs.schema_diff: schema_str += "\\nSCHEMA DIFF: " + json.dumps(obs.schema_diff)
    drift_signals = _collect_schema_drift_signals(obs)
    drift_str = ("\\nSCHEMA DRIFT SIGNALS:\\n" + "\\n".join(f"  - {s}" for s in drift_signals)) if drift_signals else ""
    actions_str = "\\n".join(f"  {a}" for a in obs.actions_taken[-5:]) or "  (none yet)"
    
    hint_str = ""
    looped_action = _detect_action_loop(obs.actions_taken)
    if looped_action: hint_str += _build_loop_hint(obs, looped_action)
    return textwrap.dedent(f"""
    STEP {step}/{obs.max_steps}
    TASK: {obs.task_id} ({obs.difficulty})
    DESCRIPTION: {obs.description}
    PIPELINE PASSED: {obs.pipeline_passed}
    LAST ACTION RESULT: {obs.last_action_result}

    DAG STRUCTURE:
    {dag_str}

    FAILING ASSERTIONS:
    {failed_str}

    PASSING ASSERTIONS: {passed_str}

    HISTORICAL RUNS:
    {hist_str}

    RECENT ACTIONS TAKEN:
    {actions_str}
    {sample_str}{schema_str}{drift_str}{hint_str}

    Respond with exactly ONE action JSON object.
    """).strip()

# ------------------------------------------------------------------ #
# OpenEnv stdout logging (spec-required — do not modify format)
# ------------------------------------------------------------------ #

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val   = error if error else "null"
    done_val    = str(done).lower()
    action_safe = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #

def run_episode(model, tokenizer, task_id: str, max_steps: int = MAX_STEPS, verbose: bool = True) -> dict:
    env = DataPipelineEnv(task_id=task_id)
    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    n_passed = 0
    n_total = 0
    pipeline_passed = False

    log_start(task=task_id, env=BENCHMARK, model=BASE_MODEL_ID)

    try:
        obs = env.reset()

        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"TASK: {task_id.upper()}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"Description: {obs.description}", file=sys.stderr)
            n_fail = len(obs.failed_assertions)
            print(f"Initial failing assertions: {n_fail}", file=sys.stderr)

        for step in range(1, max_steps + 1):
            if obs.pipeline_passed:
                if verbose:
                    print(f"\n[PASSED] Pipeline passed at step {step - 1}!", file=sys.stderr)
                break
            
            user_prompt = build_user_prompt(obs, step)
            history.append({'role': 'user', 'content': user_prompt})
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + history
            
            torch.cuda.empty_cache()
            response_text = ''
            try:
                response_text = generate(model, tokenizer, messages)
            except torch.cuda.OutOfMemoryError:
                history = history[-4:]
                messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + history
                torch.cuda.empty_cache()
                response_text = generate(model, tokenizer, messages)
            
            action = parse_llm_response(response_text)
            if action.action_type == 'run_pipeline' and not response_text.strip():
                if obs.failed_assertions:
                    action = PipelineAction(action_type='compare_schema', params={'table': obs.failed_assertions[0].table})
            
            history.append({'role': 'assistant', 'content': response_text or '{}'})
            if len(history) > 10: history = history[-10:]
            
            result = env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error: Optional[str] = getattr(obs, "last_action_error", None) or None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(action.model_dump()).replace("\n", " ")[:200],
                reward=reward,
                done=done,
                error=error,
            )

            if verbose:
                print(f"\n[Step {step}] Raw response: {response_text[:120]}", file=sys.stderr)
                print(f"[Step {step}] Action: {action.action_type}({action.params})", file=sys.stderr)
                print(f"  Reward: {reward:+.2f} | "
                      f"Passed: {len(obs.passed_assertions)}/{len(obs.failed_assertions)+len(obs.passed_assertions)} | "
                      f"Result: {obs.last_action_result[:80]}", file=sys.stderr)

            if done: break

        n_total = len(obs.failed_assertions) + len(obs.passed_assertions)
        n_passed = len(obs.passed_assertions)
        pipeline_passed = obs.pipeline_passed
        raw_score = n_passed / n_total if n_total > 0 else 0.0
        score = min(max(raw_score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD
        
        if verbose:
            print(f"\n--- Episode Summary ---", file=sys.stderr)
            print(f"  Score (assertion pass rate): {score:.2f}", file=sys.stderr)
            print(f"  Total reward:                {sum(rewards):.2f}", file=sys.stderr)
            print(f"  Steps taken:                 {steps_taken}", file=sys.stderr)
            print(f"  Pipeline passed:             {pipeline_passed}", file=sys.stderr)
        
    except Exception as exc:
        print(f"[ERROR] {task_id}: {exc}", file=sys.stderr)
    finally:
        try: env.close()
        except: pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return {
        'task_id': task_id,
        'score': round(score, 4),
        'pipeline_passed': pipeline_passed,
        'total_reward': round(sum(rewards), 4),
        'steps_taken': steps_taken,
        'assertions_passed': n_passed,
        'assertions_total': n_total,
    }

def collect_results(model_name: str, model_type: str, model, tokenizer, tasks: list):
    print(f"\n{'='*60}\nEvaluating Model: {model_name} [{model_type}]\n{'='*60}", file=sys.stderr)
    results = []
    for task_id in tasks:
        r = run_episode(model, tokenizer, task_id, max_steps=MAX_STEPS, verbose=True)
        results.append(r)
        status = '[PASSED]' if r['pipeline_passed'] else '[FAILED]'
        print(f"  {r['task_id']:8s} | score={r['score']:.2f} | reward={r['total_reward']:+.2f} | steps={r['steps_taken']:2d} | {status}", file=sys.stderr)
    avg_score = sum(r['score'] for r in results) / max(1, len(results))
    print(f"  --> Average Score: {avg_score:.4f}\n", file=sys.stderr)
    return results

def main():
    parser = argparse.ArgumentParser(description="Run Qwen data pipeline baseline/SFT/GRPO evaluations.")
    parser.add_argument('--models', type=str, nargs='+', choices=['base', 'sft', 'grpo', 'all'],
                        default=['all'], help="Which mode(s) to evaluate (base, sft, grpo, all). Space separated.")
    parser.add_argument('--tasks', type=str, nargs='+', choices=['easy', 'medium', 'hard', 'hard2', 'all'],
                        default=['all'], help="Which task(s) to run.")
    args = parser.parse_args()

    if 'all' in args.models:
        run_models = ['base', 'sft', 'grpo']
    else:
        run_models = args.models

    if 'all' in args.tasks:
        tasks = [t for t in ['easy', 'medium', 'hard', 'hard2'] if t in _AVAILABLE_TASKS]
    else:
        tasks = [t for t in args.tasks if t in _AVAILABLE_TASKS]

    all_reports = {}
    token_kwargs = {'token': HF_TOKEN} if HF_TOKEN else {}
    requires_base = ('base' in run_models) or ('sft' in run_models)

    if requires_base:
        print(f"[LOAD] Loading Base Model ({BASE_MODEL_ID}) in 16-bit...")
        # NOTE: The Kaggle script used torch.float16, matching user request to not use 4bit here
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **token_kwargs)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, 
            device_map='auto', 
            torch_dtype=torch.float16,
            **token_kwargs
        )
        base_model.eval()
        
        # 1. Base Model Eval
        if 'base' in run_models:
            all_reports['BASE'] = collect_results(BASE_MODEL_ID, "BASE", base_model, tokenizer, tasks)

        # 2. SFT Model Eval (Optional)
        if 'sft' in run_models and os.path.exists(SFT_DIR):
            print(f"[LOAD] Loading SFT Adapter from {SFT_DIR}...")
            sft_model = PeftModel.from_pretrained(base_model, SFT_DIR)
            sft_model.eval()
            all_reports['SFT'] = collect_results("Qwen SFT", "SFT", sft_model, tokenizer, tasks)
            sft_model.unload()  # Remove adapter to load GRPO

        # Unload base model to ensure enough VRAM for fully-merged models
        import gc
        del base_model
        torch.cuda.empty_cache()
        gc.collect()

    # 3. GRPO Model Eval (Optional)
    if 'grpo' in run_models:
        if not requires_base:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **token_kwargs)
        grpo_model = None
        try:
            print(f"[LOAD] Trying to load GRPO Fully-Merged from HF ({HF_REPO})...")
            grpo_model = AutoModelForCausalLM.from_pretrained(HF_REPO, device_map='auto', torch_dtype=torch.float16, **token_kwargs)
        except Exception as e:
            print(f"       HF load failed: {e}")
            try:
                print(f"[LOAD] Trying to load fully-merged local model from {LOCAL_MERGED_DIR}...")
                grpo_model = AutoModelForCausalLM.from_pretrained(LOCAL_MERGED_DIR, device_map='auto', torch_dtype=torch.float16, **token_kwargs)
            except Exception as e2:
                print(f"       Local fully-merged load failed: {e2}")
                if os.path.exists(GRPO_DIR):
                    print(f"[LOAD] Falling back to GRPO Adapter {GRPO_DIR}...")
                    base_m = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map='auto', torch_dtype=torch.float16, **token_kwargs)
                    grpo_model = PeftModel.from_pretrained(base_m, GRPO_DIR)
                
        if grpo_model is not None:
            grpo_model.eval()
            all_reports['GRPO'] = collect_results("Qwen GRPO", "GRPO", grpo_model, tokenizer, tasks)
            if hasattr(grpo_model, 'unload'):
                grpo_model.unload()
            else:
                del grpo_model
                torch.cuda.empty_cache()
                import gc
                gc.collect()

    # Final Comparison Report
    print("\\n" + "="*80)
    print("FINAL COMPARISON REPORT")
    print("="*80)
    print(f"{'Task':<10}{'Base Score':<15}{'SFT Score':<15}{'GRPO Score':<15}")
    print("-" * 80)

    for task_id in tasks:
        b_score = next((r['score'] for r in all_reports.get('BASE', []) if r['task_id'] == task_id), 0.0)
        s_score = next((r['score'] for r in all_reports.get('SFT', []) if r['task_id'] == task_id), 0.0) if 'SFT' in all_reports else '-'
        g_score = next((r['score'] for r in all_reports.get('GRPO', []) if r['task_id'] == task_id), 0.0) if 'GRPO' in all_reports else '-'
        
        b_str = f"{b_score:.2f}"
        s_str = f"{s_score:.2f}" if isinstance(s_score, float) else s_score
        g_str = f"{g_score:.2f}" if isinstance(g_score, float) else g_score
        
        print(f"{task_id:<10}{b_str:<15}{s_str:<15}{g_str:<15}")

def test_imports():
    pass

if __name__ == '__main__':
    main()
