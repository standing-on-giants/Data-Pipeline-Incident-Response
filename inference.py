"""
inference.py — Data Pipeline Incident Response Agent
=====================================================
MANDATORY environment variables:
  GROQ_API_KEY   Your Groq API key
  MODEL_NAME     (optional) override model, default: llama-3.3-70b-versatile

Usage:
  python inference.py
  python inference.py --task easy
  python inference.py --task hard --steps 25
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

# from groq import Groq as OpenAI    # alias to keep rest of file unchanged
from openai import OpenAI

# Local environment import (no WebSocket needed for local inference)
sys.path.insert(0, os.path.dirname(__file__))
from src.environment import DataPipelineEnv
from src.models import PipelineAction, PipelineObservation

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

# API_BASE_URL = None               # Groq client doesn't use base_url
# API_KEY      = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY") or "MISSING_KEY"
# MODEL_NAME   = os.getenv("MODEL_NAME") or "llama-3.3-70b-versatile"
API_BASE_URL = "http://localhost:11434/v1"
API_KEY      = "ollama"   # Ollama doesn't validate this, but the client requires it
MODEL_NAME   = os.getenv("MODEL_NAME") or "llama3"  # or mistral, phi3, etc.

MAX_STEPS   = int(os.getenv("MAX_STEPS", "20"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS  = int(os.getenv("MAX_TOKENS", "400"))

FALLBACK_ACTION = PipelineAction(action_type="run_pipeline", params={})

# ------------------------------------------------------------------ #
# System prompt
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data engineer diagnosing and fixing broken data pipelines.

You will receive the current state of a pipeline (failing assertions, DAG structure,
historical run info) and must choose ONE action to take each turn.

WORKFLOW (follow this order):
1. Read the failing assertions and historical runs to understand what broke.
2. Use read_data_sample / check_schema / compare_schema to investigate the ROOT CAUSE.
3. Apply the fix using add_data_filter or patch_transformation.
4. Call run_pipeline to verify the fix.
5. If some data is genuinely corrupted (cannot be fixed), call alert_upstream_team.

AVAILABLE ACTIONS (respond with valid JSON only):

{"action_type": "read_data_sample",
 "params": {"table": "<table_name>", "n_rows": 20}}

{"action_type": "check_schema",
 "params": {"table": "<table_name>"}}

{"action_type": "compare_schema",
 "params": {"table": "<table_name>"}}

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
- Always respond with a single JSON object (the action). No prose, no explanation.
- Do NOT apply a fix before reading the data — this will be penalised.
- Do NOT mark a failing assertion as acceptable unless it is truly non-fixable.
- Always call run_pipeline after applying a fix to see if it worked.
- If pipeline_passed is true in the observation, you are done (don't keep acting).
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
        for r in obs.historical_runs
    )

    # sample_str = ""
    # if obs.data_sample:
    #     sample_str = (
    #         "\nDATA SAMPLE (last read):\n"
    #         + json.dumps(obs.data_sample[:5], indent=2, default=str)
    #     )
    sample_str = ""
    if obs.data_sample:
        # Show first 5 rows AND any rows with null values so model can see the problem
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
    hint_str = ""
    if read_actions >= 2 and fix_actions == 0:
        hint_str = (
            "\n⚠️  HINT: You have already read the data. "
            "Stop diagnosing. Apply a fix now using add_data_filter or patch_transformation, "
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

    # Try to extract JSON block
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            l for l in lines
            if not l.startswith("```")
        ).strip()

    # Find first {...} block
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return FALLBACK_ACTION

    json_str = text[start:end]
    try:
        data = json.loads(json_str)
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
    obs = env.reset()

    # history: List[str]  = []
    # total_reward: float = 0.0
    # steps_taken: int    = 0

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
                print(f"\n✅ Pipeline passed at step {step - 1}!")
            break

        # user_prompt = build_user_prompt(obs, step)
        # messages = [
        #     {"role": "system", "content": SYSTEM_PROMPT},
        #     {"role": "user",   "content": user_prompt},
        # ]
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
        
        conversation_history.append({"role": "assistant", "content": response_text or "{}"})
        # Keep history bounded to last 10 turns to avoid token limit
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        # if verbose:
        #     print(f"\n[Step {step}] Action: {action.action_type}({action.params})")
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
        description="Data Pipeline Incident Response — baseline inference"
    )
    parser.add_argument("--task", choices=["easy", "medium", "hard", "hard2", "all"],
                        default="all", help="Which task to run (default: all)")
    parser.add_argument("--steps", type=int, default=MAX_STEPS,
                        help="Max steps per episode")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step output")
    args = parser.parse_args()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    # client = OpenAI(api_key=API_KEY)

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
        status = "✅ PASSED" if r["pipeline_passed"] else "❌ FAILED"
        print(f"  {r['task_id']:8s} | score={r['score']:.2f} | "
              f"reward={r['total_reward']:+.2f} | steps={r['steps_taken']:2d} | {status}")
        total_score += r["score"]

    avg = total_score / len(all_results) if all_results else 0.0
    print(f"\n  Average score: {avg:.4f}")

    # Machine-readable output for automated scoring
    print("\nJSON_RESULTS:", json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()