import os, sys, json, textwrap, re, torch, argparse
from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.environment import DataPipelineEnv
from src.models import PipelineAction, PipelineObservation
from src.tasks import TASKS as _AVAILABLE_TASKS

BASE_MODEL_ID  = 'Qwen/Qwen2.5-3B-Instruct'

# LoRA adapter repos on HuggingFace Hub (~200 MB each â€” no full-model download)
# Training script pushes adapters + tokenizer via model.push_to_hub()
SFT_LORA_REPO  = 'Abhinav-hf/qwen-grpo-sft-trained-16bit'  # SFT LoRA adapter repo
GRPO_LORA_REPO = 'Abhinav-hf/qwen-grpo-lora-adapters'       # GRPO LoRA adapter repo

# â”€â”€ Local fallback paths (adapter-only, NOT merged weights) 
# These are checked only if the HF Hub is unreachable.
LOCAL_SFT_LORA  = '/kaggle/working/lora_adapters_sft'
LOCAL_GRPO_LORA = '/kaggle/working/lora_adapters'

# â”€â”€ Optional: load base model in 8-bit to save VRAM (~4 GB vs ~6 GB fp16) â”€â”€
# Set USE_8BIT=True when running on T4/16 GB to avoid OOM.
# Note: 8-bit is quantization-only; torch_dtype is NOT set alongside it.
USE_8BIT = False

try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    HF_TOKEN = _s.get_secret('HF_TOKEN')
except Exception:
    HF_TOKEN = os.getenv('HF_TOKEN')

MAX_TOKENS = 1024
TEMPERATURE = 0.1
# 25 steps per episode for full evaluation; use --steps 10 for a quick sanity check.
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

KEY PATCH TYPES (you can chain multiple patches on the same step â€” they run in order):
- parse_currency: Use when a column has mixed formats like "$1,234.56" and "1234.56" and "N/A".
  It strips $, commas, casts to float, and converts N/A to NaN. Works on ANY column with "N/A" strings,
  not just currency â€” e.g. if a numeric column like impressions has "N/A" values, use parse_currency on it.
- coalesce: Use AFTER parse_currency to replace NaN/null with a default value (default is 0).
  IMPORTANT: After parse_currency, NaN values will cause value_range assertions to fail.
  You MUST chain a coalesce patch to fix this: {"action_type": "patch_transformation", "params": {"step_id": "<same_step>", "patch_type": "coalesce", "column": "<same_column>"}}
  If coalescing a denominator column (e.g. impressions used in CTR = clicks/impressions), coalescing to 0
  will cause division by zero. Instead, filter out those rows: add_data_filter with "column IS NOT NULL".
- cast_column: Use when a column needs simple numeric casting.
- dedup: Use when there are duplicate rows based on a key column.
  IMPORTANT: If a "unique" assertion is failing, the fix is ALWAYS dedup on the failing column.
  Do NOT use coalesce or add_data_filter for uniqueness failures â€” only dedup works.
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
- Do NOT apply a fix before reading the data â€” this will be penalised.
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
- If pipeline_passed is true, you are done â€” unless the task description mentions alerting an upstream team.
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
# Runner
# ------------------------------------------------------------------ #

def run_episode(model, tokenizer, task_id: str, max_steps: int = MAX_STEPS, verbose: bool = True) -> dict:
    env = DataPipelineEnv(task_id=task_id)
    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    n_passed = 0
    n_total = 0
    pipeline_passed = False

    if verbose:
        print(f"  [START] task={task_id}  max_steps={max_steps}")

    try:
        res = env.reset()
        obs = res[0] if isinstance(res, tuple) else res
        for step in range(1, max_steps + 1):
            if obs.pipeline_passed: break
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
            if isinstance(result, tuple):
                obs = result[0]
                reward = result[1]
                done = result[2]
            else:
                obs = result.observation
                reward = result.reward
                done = result.done

            rewards.append(reward or 0.0)
            steps_taken = step

            if verbose:
                n_pass_now = len(obs.passed_assertions)
                n_fail_now = len(obs.failed_assertions)
                pipe_ok = '[PIPELINE_PASSED]' if obs.pipeline_passed else ''
                print(f"    step {step:2d}/{max_steps} | action={action.action_type:<30s} "
                      f"| reward={reward:+.2f} | pass={n_pass_now} fail={n_fail_now} {pipe_ok}")

            if done: break

        n_total = len(obs.failed_assertions) + len(obs.passed_assertions)
        n_passed = len(obs.passed_assertions)
        pipeline_passed = obs.pipeline_passed
        raw_score = n_passed / n_total if n_total > 0 else 0.0
        score = min(max(raw_score, 0.01), 0.99)

    except Exception as exc:
        print(f"  [ERROR] {task_id}: {exc}", file=sys.stderr)
    finally:
        try: env.close()
        except: pass

    if verbose:
        status = '[PASSED]' if pipeline_passed else '[FAILED]'
        print(f"  [DONE]  task={task_id} score={score:.2f} steps={steps_taken} {status}")

    return {
        'task_id': task_id,
        'score': round(score, 4),
        'pipeline_passed': pipeline_passed,
        'total_reward': round(sum(rewards), 4),
        'steps_taken': steps_taken,
        'assertions_passed': n_passed,
        'assertions_total': n_total,
    }

def collect_results(model_name: str, model_type: str, model, tokenizer, tasks: list, max_steps: int = MAX_STEPS):
    import time
    print(f"\n{'='*60}\nEvaluating: {model_name} [{model_type}]  ({len(tasks)} tasks, max {max_steps} steps each)\n{'='*60}")
    results = []
    t_start = time.time()
    for task_id in tasks:
        print(f"\n  --> Task: {task_id}")
        r = run_episode(model, tokenizer, task_id, max_steps=max_steps, verbose=True)
        results.append(r)
        status = '[PASSED]' if r['pipeline_passed'] else '[FAILED]'
        elapsed = time.time() - t_start
        print(f"  RESULT {r['task_id']:8s} | score={r['score']:.2f} | reward={r['total_reward']:+.2f} "
              f"| steps={r['steps_taken']:2d} | assertions={r['assertions_passed']}/{r['assertions_total']} "
              f"| elapsed={elapsed:.0f}s | {status}")
    avg_score = sum(r['score'] for r in results) / max(1, len(results))
    print(f"\n  --> Average Score: {avg_score:.4f}  |  Total elapsed: {time.time()-t_start:.0f}s\n")
    return results

# ------------------------------------------------------------------ #
# Shared model-loading helper                                         #
# ------------------------------------------------------------------ #

def _load_base(hf_kwargs: dict) -> AutoModelForCausalLM:
    """Load Qwen2.5-3B-Instruct in fp16 or 8-bit depending on USE_8BIT flag."""
    if USE_8BIT:
        print(f"[LOAD] Base model <- {BASE_MODEL_ID}  (8-bit quantized via BitsAndBytes)")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map='auto',
            quantization_config=bnb_config,
            **hf_kwargs
        )
    else:
        print(f"[LOAD] Base model <- {BASE_MODEL_ID}  (float16)")
        return AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            device_map='auto',
            torch_dtype=torch.float16,
            **hf_kwargs
        )


def _load_lora_model(base, lora_repo: str, local_fallback: str, hf_kwargs: dict):
    """
    Apply LoRA adapters to the shared base model.

    Loading priority:
      1. HuggingFace Hub LoRA repo  (primary)
      2. local_fallback directory   (adapter-only, no merged weights)

    Returns (peft_model, tokenizer) or (None, None) on failure.
    hf_kwargs is used for HF Hub loads only; local dirs use local_files_only=True.
    """
    import os
    import gc

    tokenizer  = None
    peft_model = None

    # --- Primary: HuggingFace Hub LoRA ---
    try:
        print(f"[LOAD] LoRA tokenizer <- {lora_repo}  (latest)")
        tokenizer = AutoTokenizer.from_pretrained(lora_repo, **hf_kwargs)

        print(f"[LOAD] LoRA adapters  <- {lora_repo}  (latest)")
        peft_model = PeftModel.from_pretrained(base, lora_repo, **hf_kwargs)
    except Exception as exc:
        print(f"       HF Hub LoRA load failed: {exc}")
        tokenizer  = None
        peft_model = None

    # --- Fallback: local adapter-only directory ---
    if peft_model is None and local_fallback and os.path.exists(local_fallback):
        print(f"[LOAD] LoRA fallback  <- local dir: {local_fallback}")
        try:
            # Local dirs: no HF Hub kwargs (no token needed)
            tokenizer  = AutoTokenizer.from_pretrained(local_fallback, local_files_only=True)
            peft_model = PeftModel.from_pretrained(base, local_fallback)
        except Exception as exc:
            print(f"       Local LoRA fallback failed: {exc}")
            tokenizer  = None
            peft_model = None

    return peft_model, tokenizer


def main():
    import gc
    global USE_8BIT  # allow --use-8bit flag to override module-level default


    parser = argparse.ArgumentParser(
        description="Qwen2.5-3B  |  Base vs SFT-LoRA vs GRPO-LoRA comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python inference_qwen_comparison_GRPO_vs_og.py                   # all 3 models
          python inference_qwen_comparison_GRPO_vs_og.py --models base     # base only
          python inference_qwen_comparison_GRPO_vs_og.py --models grpo     # grpo only
          python inference_qwen_comparison_GRPO_vs_og.py --models sft grpo
          python inference_qwen_comparison_GRPO_vs_og.py --models base --tasks easy medium
          python inference_qwen_comparison_GRPO_vs_og.py --models grpo --tasks hard --steps 10
        """)
    )
    parser.add_argument(
        '--models', nargs='+',
        choices=['base', 'sft', 'grpo'],
        default=['base', 'sft', 'grpo'],
        metavar='MODEL',
        help='Models to evaluate: base sft grpo  (default: all three)'
    )
    parser.add_argument(
        '--tasks', nargs='+',
        choices=['easy', 'medium', 'hard', 'hard2'],
        default=None,
        metavar='TASK',
        help='Tasks to run: easy medium hard hard2  (default: all available)'
    )
    parser.add_argument(
        '--steps', type=int, default=MAX_STEPS,
        help=f'Max steps per episode (default: {MAX_STEPS})'
    )
    parser.add_argument(
        '--use-8bit', action='store_true', default=USE_8BIT,
        help='Load base model in 8-bit (saves ~2 GB VRAM, slower first token)'
    )
    args = parser.parse_args()

    USE_8BIT = args.use_8bit


    run_models = set(args.models)
    all_tasks  = [t for t in ['easy', 'medium', 'hard', 'hard2'] if t in _AVAILABLE_TASKS]
    tasks      = [t for t in (args.tasks or all_tasks) if t in all_tasks]
    max_steps  = args.steps

    all_reports  = {}
    token_kwargs = {'token': HF_TOKEN} if HF_TOKEN else {}
    # local_files_only=False: always check HF Hub for the latest commit
    hf_kwargs    = {**token_kwargs, 'local_files_only': False}

    print("="*80)
    print("Qwen2.5-3B  |  Inference Comparison  (LoRA adapter loading)")
    print(f"  Models to run : {sorted(run_models)}")
    print(f"  Tasks         : {tasks}")
    print(f"  Max steps/ep  : {max_steps}")
    print(f"  Quantization  : {'8-bit (BnB)' if USE_8BIT else 'float16'}")
    print(f"  Base model    : {BASE_MODEL_ID}")
    print(f"  SFT LoRA      : {SFT_LORA_REPO}")
    print(f"  GRPO LoRA     : {GRPO_LORA_REPO}")
    print("="*80)

    # ------------------------------------------------------------------ #
    # 0. SHARED BASE MODEL                                                 #
    # ------------------------------------------------------------------ #
    shared_base = None
    if 'sft' in run_models or 'grpo' in run_models or 'base' in run_models:
        try:
            shared_base = _load_base(hf_kwargs)
        except Exception as exc:
            print(f"[ERROR] Shared base model load failed: {exc}")
            shared_base = None

    # ------------------------------------------------------------------ #
    # 1. BASE MODEL  (no adapters — raw pretrained baseline)              #
    # ------------------------------------------------------------------ #
    if 'base' in run_models:
        if shared_base is not None:
            try:
                print(f"\n[LOAD] Base tokenizer <- {BASE_MODEL_ID}")
                base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **hf_kwargs)

                shared_base.eval()

                all_reports['BASE'] = collect_results(
                    BASE_MODEL_ID, "BASE", shared_base, base_tokenizer, tasks, max_steps)

                del base_tokenizer
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as exc:
                print(f"[ERROR] Base model eval failed: {exc}")
                print("        BASE column will show SKIP in the final table.")
        else:
            print("[SKIP] Base model (shared_base load failed)")
    else:
        print("[SKIP] Base model (not in --models)")

    # ------------------------------------------------------------------ #
    # 2. SFT MODEL                                                        #
    #    Apply SFT LoRA to shared base -> eval                           #
    # ------------------------------------------------------------------ #
    if 'sft' in run_models:
        if shared_base is not None:
            print(f"\n--- Loading SFT (LoRA) ---")
            sft_model, sft_tokenizer = _load_lora_model(
                base           = shared_base,
                lora_repo      = SFT_LORA_REPO,
                local_fallback = LOCAL_SFT_LORA,
                hf_kwargs      = hf_kwargs
            )
            if sft_model is not None and sft_tokenizer is not None:
                sft_model.eval()
                all_reports['SFT'] = collect_results(
                    "Qwen-SFT", "SFT", sft_model, sft_tokenizer, tasks, max_steps)
                
                # Unload adapter to restore shared base model to raw state
                shared_base = sft_model.unload()
                del sft_model, sft_tokenizer
                torch.cuda.empty_cache()
                gc.collect()
            else:
                print(f"[ERROR] SFT model could not be loaded. SFT column will show SKIP.")
        else:
            print("[SKIP] SFT model (shared_base load failed)")
    else:
        print("[SKIP] SFT model (not in --models)")

    # ------------------------------------------------------------------ #
    # 3. GRPO MODEL                                                       #
    #    Apply GRPO LoRA to shared base -> eval                          #
    # ------------------------------------------------------------------ #
    if 'grpo' in run_models:
        if shared_base is not None:
            print(f"\n--- Loading GRPO (LoRA) ---")
            grpo_model, grpo_tokenizer = _load_lora_model(
                base           = shared_base,
                lora_repo      = GRPO_LORA_REPO,
                local_fallback = LOCAL_GRPO_LORA,
                hf_kwargs      = hf_kwargs
            )
            if grpo_model is not None and grpo_tokenizer is not None:
                grpo_model.eval()
                all_reports['GRPO'] = collect_results(
                    "Qwen-GRPO", "GRPO", grpo_model, grpo_tokenizer, tasks, max_steps)
                
                shared_base = grpo_model.unload()
                del grpo_model, grpo_tokenizer
                torch.cuda.empty_cache()
                gc.collect()
            else:
                print("[ERROR] GRPO model could not be loaded. GRPO column will show SKIP.")
        else:
            print("[SKIP] GRPO model (shared_base load failed)")
    else:
        print("[SKIP] GRPO model (not in --models)")

    if shared_base is not None:
        del shared_base
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------ #
    # FINAL COMPARISON TABLE                                               #
    # Only columns for models that were actually requested are shown.      #
    # ------------------------------------------------------------------ #
    MODEL_ORDER = [('base', 'BASE', 'Base Score'),
                   ('sft',  'SFT',  'SFT Score'),
                   ('grpo', 'GRPO', 'GRPO Score')]
    active = [(flag, key, label)
              for flag, key, label in MODEL_ORDER
              if flag in run_models]

    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT")
    repo_map = {'base': BASE_MODEL_ID, 'sft': SFT_LORA_REPO, 'grpo': GRPO_LORA_REPO}
    for flag, key, _ in active:
        print(f"  {key:<5}: {repo_map[flag]}")
    print("="*80)

    col_w = 15
    header = f"{'Task':<10}" + "".join(f"{label:<{col_w}}" for _, _, label in active)
    print(header)
    print("-" * (10 + col_w * len(active)))

    all_avg = {key: 0.0 for _, key, _ in active}
    for task_id in tasks:
        row = f"{task_id:<10}"
        for _, key, _ in active:
            results_list = all_reports.get(key, [])
            if not results_list:
                cell = 'SKIP'
            else:
                sc = next((r['score'] for r in results_list if r['task_id'] == task_id), None)
                cell = f"{sc:.2f}" if sc is not None else 'N/A'
                if sc is not None:
                    all_avg[key] += sc
            row += f"{cell:<{col_w}}"
        print(row)

    print("-" * (10 + col_w * len(active)))
    n = max(len(tasks), 1)
    avg_row = f"{'AVG':<10}"
    for _, key, _ in active:
        if all_reports.get(key):
            avg_row += f"{all_avg[key]/n:.2f}{'':<{col_w-4}}"
        else:
            avg_row += f"{'N/A':<{col_w}}"
    print(avg_row)
    print("="*80)

if __name__ == '__main__':
    main()

