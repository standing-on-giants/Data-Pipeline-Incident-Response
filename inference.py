"""
inference.py — Data Pipeline Incident Response Agent
=====================================================
OpenEnv-compliant baseline inference script.

MANDATORY environment variables (set by hackathon validator):
  API_BASE_URL   Base URL for the OpenAI-compatible LLM endpoint
  MODEL_NAME     Model to use (e.g. meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN       HuggingFace token used as the API key

Optional overrides:
  MAX_STEPS      Max actions per episode (default: 20)
  TEMPERATURE    Sampling temperature (default: 0.1)
  MAX_TOKENS     Max tokens per completion (default: 600)

Usage:
  python inference.py
  python inference.py --task easy
  python inference.py --task hard2 --steps 30
  python inference.py --task all --quiet
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Local environment import (no WebSocket needed for local inference)
sys.path.insert(0, os.path.dirname(__file__))
from src.environment import DataPipelineEnv
from src.models import PipelineAction, PipelineObservation

# ------------------------------------------------------------------ #
# Configuration — all values read from environment variables per spec
# ------------------------------------------------------------------ #

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "MISSING_KEY"
MODEL_NAME   = os.getenv("MODEL_NAME", "llama3")

BENCHMARK   = "data_pipeline_incident_response"
MAX_STEPS   = int(os.getenv("MAX_STEPS", "20"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "600"))

SUCCESS_SCORE_THRESHOLD = 0.1

# Safe fallback: compare_schema is a read-only diagnostic action
# (no blind-fix penalty, no state mutation). run_pipeline as fallback
# would trigger the -0.5 penalty if applied before reading data.
FALLBACK_ACTION = PipelineAction(
    action_type="compare_schema",
    params={"table": "raw_orders"},
)

# ------------------------------------------------------------------ #
# OpenEnv stdout logging  — DO NOT MODIFY: format parsed by validator
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
# System prompt
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data engineer diagnosing and fixing broken data pipelines.

You will receive the current state of a pipeline (failing assertions, DAG structure,
historical run info) and must choose ONE action to take each turn.

WORKFLOW — always follow this order:
1. Read the failing assertions to understand what broke.
2. Use read_data_sample / check_schema / compare_schema to diagnose the ROOT CAUSE.
3. If compare_schema shows column renames or new columns → call handle_drift first.
4. Apply the fix using add_data_filter or patch_transformation.
5. Call run_pipeline to verify.
6. If data is genuinely corrupted and unfixable, call alert_upstream_team.

CRITICAL SCHEMA DRIFT RULE:
If compare_schema reveals a column was renamed (e.g. spend → total_spend), call:
  handle_drift with strategy="resolve_column_rename" BEFORE any patch.
If compare_schema shows new/removed columns, call handle_drift with strategy="detect" first.

AVAILABLE ACTIONS (respond with valid JSON only):

{"action_type": "read_data_sample",
 "params": {"table": "<table_name>", "n_rows": 20}}

{"action_type": "check_schema",
 "params": {"table": "<table_name>"}}

{"action_type": "compare_schema",
 "params": {"table": "<table_name>"}}

{"action_type": "handle_drift",
 "params": {"strategy": "<detect|resolve_column_rename|numeric_format|null_fill|type_cast|join_key_prefix|filter_invalid|alert_upstream>",
            "table": "<table_name>",
            "old_column": "<original_col_name>",
            "new_column": "<renamed_col_name>"}}

{"action_type": "run_quality_assertion",
 "params": {"assertion_id": "<e.g. A1>"}}

{"action_type": "add_data_filter",
 "params": {"step_id": "<step_id>", "filter_condition": "<e.g. user_id IS NOT NULL>"}}

{"action_type": "patch_transformation",
 "params": {"step_id": "<step_id>",
            "patch_type": "<cast_column|coalesce|dedup|parse_currency|strip_prefix>",
            "column": "<column_name>"}}

{"action_type": "backfill_partition",
 "params": {"date": "<YYYY-MM-DD>"}}

{"action_type": "alert_upstream_team",
 "params": {"team": "<team_name>",
            "issue_description": "<what is wrong>"}}

{"action_type": "mark_acceptable",
 "params": {"assertion_id": "<id>", "reason": "<reason>"}}

{"action_type": "run_pipeline", "params": {}}

RULES:
- Always respond with a SINGLE JSON object. No prose. No markdown.
- NEVER apply a fix (filter/patch) before reading the data — heavy penalty applies.
- NEVER mark a failing assertion as acceptable unless it is truly non-fixable.
- Always call run_pipeline after applying a fix to verify it worked.
- If pipeline_passed is true, stop immediately — do not keep acting.
- If a column is missing after run_pipeline, call compare_schema before patching.
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
        f"  {n.step_id}: {n.input_table} -> {n.output_table}"
        + (f" | filters: {n.applied_filters}" if n.applied_filters else "")
        + (f" | patches: {n.applied_patches}" if n.applied_patches else "")
        for n in obs.dag_structure
    )

    hist_str = "\n".join(
        f"  {r.date}: {r.status} ({r.row_count} rows)"
        for r in obs.historical_runs
    )

    sample_str = ""
    if obs.data_sample:
        sample_rows = obs.data_sample[:5]
        null_rows   = [r for r in obs.data_sample if any(v is None for v in r.values())]
        sample_str  = (
            "\nDATA SAMPLE (first 5 rows):\n"
            + json.dumps(sample_rows, indent=2, default=str)
        )
        if null_rows:
            sample_str += (
                f"\nROWS WITH NULL VALUES ({len(null_rows)} found):\n"
                + json.dumps(null_rows[:5], indent=2, default=str)
            )

    schema_str = ""
    if obs.current_schema:
        schema_str = "\nCURRENT SCHEMA: " + json.dumps(obs.current_schema)
    if obs.historical_schema:
        schema_str += "\nHISTORICAL SCHEMA: " + json.dumps(obs.historical_schema)
    if obs.schema_diff:
        schema_str += "\nSCHEMA DIFF (new/removed/changed columns): " + json.dumps(obs.schema_diff)

    actions_str = "\n".join(f"  {a}" for a in obs.actions_taken[-6:]) or "  (none yet)"

    # Adaptive hint: if agent keeps reading without fixing
    read_count = sum(1 for a in obs.actions_taken
                     if any(k in a for k in ("read_data_sample", "check_schema", "compare_schema")))
    fix_count  = sum(1 for a in obs.actions_taken
                     if any(k in a for k in ("add_data_filter", "patch_transformation", "handle_drift")))
    hint_str = ""
    if read_count >= 2 and fix_count == 0:
        hint_str = (
            "\n[HINT] You have already read the data. Stop diagnosing. "
            "Apply a fix now (add_data_filter, patch_transformation, or handle_drift), "
            "then call run_pipeline."
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
    """Extract and validate a PipelineAction from the model's response text."""
    if not text:
        return FALLBACK_ACTION

    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(l for l in lines if not l.startswith("```")).strip()

    # Find first {...} block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return FALLBACK_ACTION

    try:
        data = json.loads(text[start:end])
        return PipelineAction(**data)
    except Exception:
        return FALLBACK_ACTION


# ------------------------------------------------------------------ #
# Single episode runner
# ------------------------------------------------------------------ #

def run_episode(
    client: OpenAI,
    task_id: str,
    max_steps: int = MAX_STEPS,
    verbose: bool = True,
) -> Dict[str, Any]:
    env = DataPipelineEnv(task_id=task_id)

    history:         List[Dict[str, str]] = []
    rewards:         List[float]          = []
    steps_taken:     int                  = 0
    score:           float                = 0.0
    success:         bool                 = False
    n_passed:        int                  = 0
    n_total:         int                  = 0
    pipeline_passed: bool                 = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"TASK: {task_id.upper()}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            print(f"Description: {obs.description}", file=sys.stderr)
            print(f"Initial failing assertions: {len(obs.failed_assertions)}", file=sys.stderr)

        for step in range(1, max_steps + 1):
            if obs.pipeline_passed:
                if verbose:
                    print(f"\n[PASSED] Pipeline passed at step {step - 1}!", file=sys.stderr)
                break

            user_prompt = build_user_prompt(obs, step)
            history.append({"role": "user", "content": user_prompt})

            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

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
                print(f"  [Step {step}] API error: {exc}. Using fallback.", file=sys.stderr, flush=True)
                response_text = ""

            action = parse_llm_response(response_text)

            history.append({"role": "assistant", "content": response_text or "{}"})
            # Keep history bounded to avoid token limit overflow
            if len(history) > 20:
                history = history[-20:]

            result = env.step(action)
            obs    = result.observation
            reward = result.reward or 0.0
            done   = result.done
            error: Optional[str] = None

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
                print(f"\n[Step {step}] Action: {action.action_type}({action.params})", file=sys.stderr)
                print(
                    f"  Reward: {reward:+.2f} | "
                    f"Passed: {len(obs.passed_assertions)}/{len(obs.failed_assertions)+len(obs.passed_assertions)} | "
                    f"Result: {obs.last_action_result[:100]}",
                    file=sys.stderr,
                )

            if done:
                break

        # Final score: fraction of assertions passing, clipped per OpenEnv spec
        n_total  = len(obs.failed_assertions) + len(obs.passed_assertions)
        n_passed = len(obs.passed_assertions)
        pipeline_passed = obs.pipeline_passed
        raw_score = n_passed / n_total if n_total > 0 else 0.0
        score     = min(max(raw_score, 0.01), 0.99)
        success   = score >= SUCCESS_SCORE_THRESHOLD

        if verbose:
            print(f"\n--- Episode Summary ---", file=sys.stderr)
            print(f"  Score (assertion pass rate): {score:.2f}", file=sys.stderr)
            print(f"  Total reward:                {sum(rewards):.2f}", file=sys.stderr)
            print(f"  Steps taken:                 {steps_taken}", file=sys.stderr)
            print(f"  Pipeline passed:             {pipeline_passed}", file=sys.stderr)

    except Exception as exc:
        print(f"[ERROR] Episode error task={task_id}: {exc}", file=sys.stderr, flush=True)

    finally:
        try:
            env.close()
        except AttributeError:
            pass

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":            task_id,
        "score":              round(score, 4),
        "success":            success,
        "pipeline_passed":    pipeline_passed,
        "total_reward":       round(sum(rewards), 4),
        "steps_taken":        steps_taken,
        "assertions_passed":  n_passed,
        "assertions_total":   n_total,
    }


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Data Pipeline Incident Response — OpenEnv baseline inference"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "hard2", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="Max steps per episode")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step stderr output")
    args = parser.parse_args()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

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

    # Summary to stderr — keeps stdout clean for the OpenEnv spec parser
    print("\n" + "=" * 60, file=sys.stderr)
    print("FINAL SCORES", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    total_score = 0.0
    for r in all_results:
        status = "[PASSED]" if r["pipeline_passed"] else "[FAILED]"
        print(
            f"  {r['task_id']:8s} | score={r['score']:.2f} | "
            f"reward={r['total_reward']:+.2f} | steps={r['steps_taken']:2d} | {status}",
            file=sys.stderr,
        )
        total_score += r["score"]

    avg = total_score / len(all_results) if all_results else 0.0
    print(f"\n  Average score: {avg:.4f}", file=sys.stderr)
    print("\nJSON_RESULTS:", json.dumps(all_results, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
