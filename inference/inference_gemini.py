"""
new_inference.py — Data Pipeline Incident Response Agent  (Gemini 2.5 Flash via OpenAI APICompat)
================================================================================================
Uses Google Gemini 2.5 Flash via the standard OpenAI Python SDK using the OpenAI compatibility endpoint.

MANDATORY environment variables:
  GEMINI_API_KEY   Your Google Gemini API key

Usage:
  python new_inference.py
  python new_inference.py --task easy
  python new_inference.py --task hard --steps 25
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env with override so it always wins over existing env vars
load_dotenv(override=True)

from openai import OpenAI

# Local environment import
sys.path.insert(0, os.path.dirname(__file__))
from src.environment import DataPipelineEnv
from src.models import PipelineAction, PipelineObservation

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

# Point OpenAI client to Gemini's compatibility endpoint!
API_BASE_URL   = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "MISSING_KEY"
MODEL_NAME     = os.getenv("MODEL_NAME") or "gemini-2.5-flash"

MAX_STEPS      = int(os.getenv("MAX_STEPS", "30"))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "1024"))

FALLBACK_ACTION = PipelineAction(action_type="run_pipeline", params={})

# ------------------------------------------------------------------ #
# System prompt
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data engineer diagnosing and fixing broken data pipelines.

You will receive the current state of a pipeline (failing assertions, DAG structure,
historical run info) and must choose ONE action to take each turn.

WORKFLOW (follow this order strictly):
1. FIRST: read_data_sample on the raw input table(s) to see what the data looks like.
2. THEN: Use check_schema or compare_schema if a type or schema issue is suspected.
3. THEN: Apply the RIGHT fix using add_data_filter or patch_transformation.
4. THEN: Call run_pipeline to verify the fix.
5. ONLY AFTER fixing what you can: If some data is genuinely corrupted (e.g. "N/A" values
   that cannot be parsed), call alert_upstream_team.
6. If assertions are still failing after run_pipeline, investigate more and apply
   additional fixes. Do NOT just call run_pipeline again without changing something.

AVAILABLE ACTIONS (respond with ONLY a JSON object, no markdown, no prose):

{"action_type": "read_data_sample", "params": {"table": "<table_name>", "n_rows": 20}}
{"action_type": "check_schema", "params": {"table": "<table_name>"}}
{"action_type": "compare_schema", "params": {"table": "<table_name>"}}
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
- strip_prefix: Use when column values have a spurious prefix like "CMP_" that needs removal.
  Params: step_id, column. Optionally "prefix" (default "CMP_"). After stripping, chain cast_column
  if the underlying value should be numeric.

UPSTREAM TEAM NAMING:
- Team names are always lowercase snake_case. Examples: meta_ads_api_team, data_engineering, vendor_support.
- If the description mentions "Meta", "Graph API", or "Meta Ads", the team is likely "meta_ads_api_team".

RULES:
- RESPOND WITH ONLY A JSON OBJECT. No markdown fences, no explanation, no prose.
- Do NOT call run_pipeline unless you applied a filter or patch since the last run.
- Do NOT apply a fix before reading the data — this will be penalised.
- NEVER use mark_acceptable. It always results in a heavy penalty. Instead, fix the data.
- After parse_currency, ALWAYS chain coalesce on the same column to handle NaN values before calling run_pipeline.
- If a computed column (like CTR) has a value_range failure, check ALL input columns in its formula.
  For example, if CTR = clicks/impressions and impressions has "N/A" strings, you must fix impressions
  with parse_currency first, then filter out null rows, before the computed column can produce valid values.
- If a joined output table has 0 rows (row_count assertion failing), the join keys likely don't match.
  Use compare_schema on the input tables to detect type/format drifts like string vs int, or unwanted
  prefixes on the join key. Apply strip_prefix + cast_column to align the keys.
- If pipeline_passed is true, you are done — unless the task description mentions alerting an upstream team.
""").strip()


# ------------------------------------------------------------------ #
# Prompt builder
# ------------------------------------------------------------------ #

def build_user_prompt(obs: PipelineObservation, step: int) -> str:
    failed_str = "\n".join(
        f"  [{r.assertion_id}] {r.assertion_type} on {r.table}"
        f"({r.column or 'N/A'}): {r.actual}"
        for r in obs.failed_assertions
    ) or "  (none — all passing!)"

    passed_str = ", ".join(r.assertion_id for r in obs.passed_assertions) or "none"

    dag_str = "\n".join(
        f"  {n.step_id}: {n.input_table} → {n.output_table}"
        + (f" | filters: {n.applied_filters}" if n.applied_filters else "")
        + (f" | patches: {n.applied_patches}" if n.applied_patches else "")
        for n in obs.dag_structure
    )

    hist_str = "\n".join(
        f"  {r.date}: {r.status} ({r.row_count} rows)"
        for r in obs.historical_runs[-2:]
    )

    sample_str = ""
    if obs.data_sample:
        sample_rows = obs.data_sample[:5]
        null_rows = [r for r in obs.data_sample if any(v is None for v in r.values())]
        if null_rows:
            sample_str = (
                "\nDATA SAMPLE (first 5 rows):\n"
                + json.dumps(sample_rows, indent=2, default=str)
                + f"\nROWS WITH NULL VALUES ({len(null_rows)} found):\n"
                + json.dumps(null_rows[:5], indent=2, default=str)
            )
        else:
            sample_str = (
                "\nDATA SAMPLE (first 5 rows):\n"
                + json.dumps(sample_rows, indent=2, default=str)
            )

    schema_str = ""
    if obs.current_schema:
        schema_str = "\nCURRENT SCHEMA: " + json.dumps(obs.current_schema)
    if obs.schema_diff:
        schema_str += "\nSCHEMA DIFF vs HISTORICAL: " + json.dumps(obs.schema_diff)

    actions_str = "\n".join(f"  {a}" for a in obs.actions_taken[-5:]) or "  (none yet)"

    # Detect if the model has been reading without acting
    read_actions = sum(1 for a in obs.actions_taken if "read_data_sample" in a or "check_schema" in a)
    fix_actions  = sum(1 for a in obs.actions_taken if "add_data_filter" in a or "patch_transformation" in a)
    run_actions  = sum(1 for a in obs.actions_taken if "run_pipeline" in a)
    mark_actions = sum(1 for a in obs.actions_taken if "mark_acceptable" in a)
    parse_done   = any("parse_currency" in a for a in obs.actions_taken)
    coalesce_done = any("coalesce" in a for a in obs.actions_taken)
    hint_str = ""
    if read_actions >= 2 and fix_actions == 0:
        hint_str = (
            "\n[HINT]: You have already read the data. "
            "Stop diagnosing. Apply a fix now using add_data_filter or patch_transformation, "
            "then call run_pipeline."
        )
    # Detect value_range failing after parse_currency was applied (NaN issue)
    value_range_failing = any(
        r.assertion_type == "value_range" and "non-numeric" in r.actual
        for r in obs.failed_assertions
    )
    if parse_done and not coalesce_done and value_range_failing:
        hint_str += (
            "\n[CRITICAL]: A value_range assertion is STILL failing because parse_currency converts "
            "unparseable values (like 'N/A') to NaN, and NaN counts as out-of-range. "
            "You MUST apply a coalesce patch to replace NaN with 0 on the same column and step "
            "where you applied parse_currency."
        )
    # Detect mark_acceptable abuse
    if mark_actions >= 1:
        hint_str += (
            "\n[WARNING]: NEVER use mark_acceptable again. It gives a -1.0 penalty every time. "
            "Instead, apply a coalesce patch to fix NaN values, then run_pipeline."
        )
    # Detect run_pipeline loops
    recent = obs.actions_taken[-3:]
    recent_runs = sum(1 for a in recent if "run_pipeline" in a)
    if recent_runs >= 2 and not obs.pipeline_passed:
        hint_str += (
            "\n[CRITICAL]: You have called run_pipeline multiple times with no progress. "
            "You MUST apply a fix (patch_transformation or add_data_filter) before calling run_pipeline again."
        )


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
    {sample_str}{schema_str}{hint_str}

    Respond with exactly ONE action JSON object.
    """).strip()


# ------------------------------------------------------------------ #
# Action parser
# ------------------------------------------------------------------ #

def parse_llm_response(text: str) -> PipelineAction:
    """Extract and validate a PipelineAction from the model's response text.
    Handles truncated JSON by attempting to close incomplete objects."""
    if not text:
        return FALLBACK_ACTION

    text = text.strip()
    # Strip markdown code fences if present
    if "```" in text:
        lines = text.split("\n")
        text = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        ).strip()

    # Find first {...} block
    start = text.find("{")
    if start == -1:
        return FALLBACK_ACTION

    end = text.rfind("}") + 1

    if end > start:
        json_str = text[start:end]
        try:
            data = json.loads(json_str)
            return PipelineAction(**data)
        except Exception:
            pass

    # JSON might be truncated — attempt to repair
    fragment = text[start:]
    repaired = _try_repair_json(fragment)
    if repaired:
        try:
            return PipelineAction(**repaired)
        except Exception:
            pass

    return FALLBACK_ACTION


def _try_repair_json(fragment: str) -> Optional[dict]:
    """Try to repair truncated JSON from LLM output."""
    # Common case: {"action_type": "patch_transformation", "params": {"step_id": "transform_crm", ...  (truncated)
    # Strategy: try closing brackets progressively
    for suffix in ['"}}', '}}', '}"}}', '"}}', '}']:
        try:
            data = json.loads(fragment + suffix)
            if isinstance(data, dict) and "action_type" in data:
                return data
        except Exception:
            continue

    # Try extracting just the action_type to make a minimal valid action
    import re
    at_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', fragment)
    if not at_match:
        return None

    action_type = at_match.group(1)
    params = {}

    # Try to extract params fields
    step_match = re.search(r'"step_id"\s*:\s*"([^"]+)"', fragment)
    patch_match = re.search(r'"patch_type"\s*:\s*"([^"]+)"', fragment)
    col_match = re.search(r'"column"\s*:\s*"([^"]+)"', fragment)
    table_match = re.search(r'"table"\s*:\s*"([^"]+)"', fragment)
    filter_match = re.search(r'"filter_condition"\s*:\s*"([^"]+)"', fragment)
    team_match = re.search(r'"team"\s*:\s*"([^"]+)"', fragment)
    issue_match = re.search(r'"issue_description"\s*:\s*"([^"]+)"', fragment)
    n_rows_match = re.search(r'"n_rows"\s*:\s*(\d+)', fragment)

    if step_match:
        params["step_id"] = step_match.group(1)
    if patch_match:
        params["patch_type"] = patch_match.group(1)
    if col_match:
        params["column"] = col_match.group(1)
    if table_match:
        params["table"] = table_match.group(1)
    if filter_match:
        params["filter_condition"] = filter_match.group(1)
    if team_match:
        params["team"] = team_match.group(1)
    if issue_match:
        params["issue_description"] = issue_match.group(1)
    if n_rows_match:
        params["n_rows"] = int(n_rows_match.group(1))

    try:
        return {"action_type": action_type, "params": params}
    except Exception:
        return None


# ------------------------------------------------------------------ #
# Single episode runner
# ------------------------------------------------------------------ #

def run_episode(
    client: OpenAI,
    task_id: str,
    max_steps: int = MAX_STEPS,
    verbose: bool = True,
) -> Dict[str, Any]:
    env = DataPipelineEnv(task_id=task_id, max_steps=max_steps)
    obs = env.reset()

    conversation_history: List[Dict[str, str]] = []
    total_reward: float = 0.0
    steps_taken: int    = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"TASK: {task_id.upper()}")
        print(f"{'='*60}")
        print(f"Description: {obs.description}")
        n_fail = len(obs.failed_assertions)
        print(f"Initial failing assertions: {n_fail}")

    for step in range(1, max_steps + 1):
        if obs.pipeline_passed:
            if verbose:
                print(f"\n[PASSED] Pipeline passed at step {step - 1}!")
            break

        user_prompt = build_user_prompt(obs, step)
        conversation_history.append({"role": "user", "content": user_prompt})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ] + conversation_history

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            if verbose:
                print(f"  [Step {step}] API error: {exc}. Using fallback.")
            response_text = ""

        action = parse_llm_response(response_text)

        # Smart fallback: if model returned empty/garbage and action is run_pipeline,
        # convert the wasted step into a diagnostic compare_schema instead
        if action.action_type == "run_pipeline" and not response_text.strip():
            if obs.failed_assertions:
                target_table = obs.failed_assertions[0].table
                action = PipelineAction(
                    action_type="compare_schema",
                    params={"table": target_table}
                )
        
        conversation_history.append({"role": "assistant", "content": response_text or "{}"})
        # Keep history bounded to last 20 messages to avoid token limit
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        if verbose:
            print(f"\n[Step {step}] Raw response: {response_text[:120]}")
            print(f"[Step {step}] Action: {action.action_type}({action.params})")

        result = env.step(action)
        obs    = result.observation
        total_reward += result.reward
        steps_taken   = step

        if verbose:
            print(f"  Reward: {result.reward:+.2f} | "
                  f"Passed: {len(obs.passed_assertions)}/{len(obs.failed_assertions)+len(obs.passed_assertions)} | "
                  f"Result: {obs.last_action_result[:80]}")

        if result.done:
            break

    # Final score: fraction of assertions passing
    n_total  = len(obs.failed_assertions) + len(obs.passed_assertions)
    n_passed = len(obs.passed_assertions)
    score    = n_passed / n_total if n_total > 0 else 0.0

    if verbose:
        print(f"\n--- Episode Summary ---")
        print(f"  Score (assertion pass rate): {score:.2f}")
        print(f"  Total reward:                {total_reward:.2f}")
        print(f"  Steps taken:                 {steps_taken}")
        print(f"  Pipeline passed:             {obs.pipeline_passed}")

    return {
        "task_id":        task_id,
        "score":          round(score, 4),
        "pipeline_passed": obs.pipeline_passed,
        "total_reward":   round(total_reward, 4),
        "steps_taken":    steps_taken,
        "assertions_passed": n_passed,
        "assertions_total":  n_total,
    }


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Data Pipeline Incident Response — Gemini 2.5 Flash via OpenAI API Compat"
    )
    parser.add_argument("--task", choices=["easy", "medium", "hard", "hard2", "all"],
                        default="all", help="Which task to run (default: all)")
    parser.add_argument("--steps", type=int, default=MAX_STEPS,
                        help="Max steps per episode")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step output")
    args = parser.parse_args()

    # Initialise the OpenAI client pointing to Google's API compatibility endpoint
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=GEMINI_API_KEY
    )

    tasks = ["easy", "medium", "hard", "hard2"] if args.task == "all" else [args.task]

    all_results = []
    for task_id in tasks:
        result = run_episode(
            client=client,
            task_id=task_id,
            max_steps=args.steps,
            verbose=not args.quiet,
        )
        all_results.append(result)

    print("\n" + "="*60)
    print("FINAL SCORES")
    print("="*60)
    total_score = 0.0
    for r in all_results:
        status = "[PASSED]" if r["pipeline_passed"] else "[FAILED]"
        print(f"  {r['task_id']:8s} | score={r['score']:.2f} | "
              f"reward={r['total_reward']:+.2f} | steps={r['steps_taken']:2d} | {status}")
        total_score += r["score"]

    avg = total_score / len(all_results) if all_results else 0.0
    print(f"\n  Average score: {avg:.4f}")

    # Machine-readable output for automated scoring
    print("\nJSON_RESULTS:", json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
