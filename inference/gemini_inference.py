"""
gemini_inference.py — Legal Contract Review Agent (Gemini)
===========================================

Environment variables (set before running):
    API_BASE_URL   LLM endpoint.  Default: https://generativelanguage.googleapis.com/v1beta/openai/
    MODEL_NAME     Model name.    Default: gemini-2.5-flash
    GEMINI_API_KEY API key.       Required to authenticate with Gemini.
    LOCAL_IMAGE_NAME  Docker image name if using from_docker_image() (optional)

STDOUT FORMAT (OpenEnv spec — do not change):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    python gemini_inference.py                  # run all three tasks
    python gemini_inference.py --task easy
    python gemini_inference.py --task hard --steps 35
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env without override so the server's injected environment variables always win
load_dotenv(override=False)

# ── OpenAI client pointed at Gemini ──
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from src.environment import LegalContractEnv
from src.models import ContractAction, ContractObservation

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

# Gemini's OpenAI-compatible endpoint
API_BASE_URL     = os.getenv("API_BASE_URL",  "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME       = os.getenv("MODEL_NAME",    "gemini-2.5-flash")
API_KEY          = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY", "MISSING_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # only needed for docker-based envs

BENCHMARK        = "legal_contract_review"
MAX_STEPS        = int(os.getenv("MAX_STEPS",    "100"))
TEMPERATURE      = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS       = int(os.getenv("MAX_TOKENS",   "1024"))

SUCCESS_SCORE_THRESHOLD = 0.1   # score in [0, 1] to count as success

FALLBACK_ACTION = ContractAction(action_type="summarize", params={})


# ------------------------------------------------------------------ #
# System prompt (unchanged from original)
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert legal analyst reviewing contracts for missing clauses, risky terms,
and non-standard language on behalf of the client.

You receive the current review state and must choose ONE action per turn.
Respond with exactly ONE JSON object — no prose, no explanation.

WORKFLOW (follow in order):
1. Read every section with read_section before flagging anything in it.
2. Flag risky / non-standard clauses with flag_clause.
3. Flag completely absent but required clauses with mark_missing.
4. For every flag, follow up with suggest_redline to propose replacement language.
5. Approve clean sections with approve_section.
6. When all sections reviewed and all issues flagged, call summarize.

AVAILABLE ACTIONS:

{"action_type": "read_section", "params": {"section": "<section_name>"}}

{"action_type": "flag_clause",
 "params": {"section": "<section_name>",
            "clause_id": "<DESCRIPTIVE id e.g. 'unlimited_indemnity', 'auto_renewal_buried'>",
            "risk_level": "<low|medium|critical>",
            "reason": "<concrete explanation>"}}

{"action_type": "mark_missing",
 "params": {"section": "<section_name>",
            "clause_id": "<DESCRIPTIVE id e.g. 'liability_cap', 'sla_uptime', 'survival_clause'>",
            "risk_level": "<low|medium|critical>",
            "reason": "<what is absent and why it matters>"}}

{"action_type": "suggest_redline",
 "params": {"clause_id": "<same id as the flag>",
            "replacement_text": "<standard language to replace/add the clause>"}}

{"action_type": "approve_section", "params": {"section": "<section_name>"}}

{"action_type": "summarize", "params": {}}

RULES:
- ONE JSON object per turn. No prose.
- Never flag before reading the section — penalty applied.
- Use DESCRIPTIVE clause_id values, not generic C1/C2/M1.
- Only flag genuine issues — false positives are penalised.
- For M&A: 1% tipping basket and 18-month rep survival are MARKET STANDARD — do not flag them.
- ALWAYS call summarize when done. Do NOT hit the step limit without summarizing.
- If STEPS REMAINING <= 3, call summarize immediately.
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
# Prompt builder (unchanged from original)
# ------------------------------------------------------------------ #

def build_user_prompt(obs: ContractObservation, step: int, max_steps: int) -> str:
    steps_remaining = max_steps - step

    sections_str = "\n".join(
        f"  [{s.section_name}] {'READ' if s.read else 'UNREAD'}"
        + (" | approved" if s.approved else "")
        + (f" | flags: {s.flags_count}" if s.flags_count else "")
        for s in obs.section_statuses
    )

    risky_flags   = [f for f in obs.flags if f.flag_type == "risky"]
    missing_flags = [f for f in obs.flags if f.flag_type == "missing"]

    flags_str = "\n".join(
        f"  [{f.clause_id}] {f.risk_level.upper()} in [{f.section}]: {f.reason[:120]}"
        for f in risky_flags
    ) or "  (none yet)"

    missing_str = "\n".join(
        f"  [{f.clause_id}] in [{f.section}]: {f.reason[:120]}"
        for f in missing_flags
    ) or "  (none yet)"

    redlined = [f for f in obs.flags if getattr(f, "redline_suggested", False)]
    redlines_str = "\n".join(
        f"  [{f.clause_id}] — redline submitted" for f in redlined
    ) or "  (none yet)"

    section_text_str = ""
    if obs.current_section_text and obs.current_section_name:
        section_text_str = (
            f"\nLAST READ SECTION — [{obs.current_section_name}]:\n"
            + obs.current_section_text
        )

    actions_str = "\n".join(f"  {a}" for a in obs.actions_taken[-12:]) or "  (none yet)"

    hints: List[str] = []
    read_actions = sum(1 for a in obs.actions_taken if "read_section" in a)
    flag_actions = sum(
        1 for a in obs.actions_taken if "flag_clause" in a or "mark_missing" in a
    )
    if read_actions >= 3 and flag_actions == 0:
        hints.append(
            "WARNING: Several sections read but nothing flagged yet. "
            "Use flag_clause / mark_missing if issues exist, or approve_section if clean."
        )
    unread = [s for s in obs.section_statuses if not s.read]
    flags_done = len(obs.flags) > 0
    redlines_done = any(getattr(f, "redline_suggested", False) for f in obs.flags)
    if not unread and flags_done and redlines_done and not obs.done:
        hints.append("ALL SECTIONS READ AND FLAGGED. Call summarize to finalise the review.")
    elif not unread and not flags_done and not obs.done:
        hints.append("ALL SECTIONS READ. Now flag risky clauses and missing provisions before summarizing.")
    if steps_remaining <= 3 and not obs.done:
        hints.append(
            f"URGENT — ONLY {steps_remaining} STEPS REMAINING. "
            "Call summarize NOW to avoid hitting the step limit without a graded result."
        )

    hints_str = ("\n" + "\n".join(hints)) if hints else ""

    return textwrap.dedent(f"""
    STEP {step}/{max_steps}  |  STEPS REMAINING: {steps_remaining}
    # TASK: {obs.task_id} ({obs.difficulty})
    TASK: {obs.task_id}
    CONTRACT: {obs.contract_title}
    DESCRIPTION: {obs.description}
    REVIEW COMPLETE: {obs.done}
    FAULTS FOUND SO FAR: {obs.faults_found_so_far}/{obs.total_faults_in_contract}
    LAST ACTION RESULT: {obs.last_action_result}

    CONTRACT SECTIONS:
    {sections_str}

    FLAGS RAISED (risky clauses):
    {flags_str}

    MISSING CLAUSES MARKED:
    {missing_str}

    REDLINES SUBMITTED:
    {redlines_str}

    RECENT ACTIONS:
    {actions_str}
    {section_text_str}

    {hints_str}

    Respond with exactly ONE action JSON object.
    """).strip()

# ------------------------------------------------------------------ #
# Action parser (unchanged from original)
# ------------------------------------------------------------------ #

def parse_llm_response(text: str) -> ContractAction:
    if not text:
        return FALLBACK_ACTION
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return FALLBACK_ACTION
    try:
        data = json.loads(text[start:end])
        return ContractAction(**data)
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
    env = LegalContractEnv(task_id=task_id, max_steps=max_steps)

    history:     List[Dict[str, str]] = []
    rewards:     List[float]          = []
    steps_taken: int                  = 0
    score:       float                = 0.0
    success:     bool                 = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"TASK: {task_id.upper()} — {obs.contract_title}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            steps_remaining = max_steps - step

            # Force summarize when budget is nearly exhausted
            if steps_remaining <= 1 and not obs.done:
                action        = ContractAction(action_type="summarize", params={})
                response_text = json.dumps({"action_type": "summarize", "params": {}})
            else:
                user_prompt = build_user_prompt(obs, step, max_steps)
                history.append({"role": "user", "content": user_prompt})
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

                try:
                    # ── Official OpenAI API call ──────────────────────
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = (completion.choices[0].message.content or "").strip()
                except Exception as exc:
                    print(f"[DEBUG] LLM error step {step}: {exc}", file=sys.stderr, flush=True)
                    response_text = ""

                history.append({"role": "assistant", "content": response_text or "{}"})

                # Keep conversation window bounded (memory budget)
                if len(history) > 40:
                    history = history[-40:]

            action = parse_llm_response(response_text)

            result  = env.step(action)
            obs     = result.observation
            reward  = result.reward or 0.0
            done    = result.done
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
                print(
                    f"  [Step {step}] {action.action_type} | reward={reward:+.2f} | "
                    f"caught={obs.faults_found_so_far}/{obs.total_faults_in_contract}",
                    file=sys.stderr,
                )

            if done:
                break

        # Extract real grader score from the summarize result message
        import re as _re
        _score_match = _re.search(r'Score=(\d+\.\d+)', obs.last_action_result)
        if _score_match:
            score = float(_score_match.group(1))
        else:
            # Fallback: raw recall (only if summarize was never called)
            n_total  = obs.total_faults_in_contract
            n_caught = obs.faults_found_so_far
            score = n_caught / n_total if n_total > 0 else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error task={task_id}: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id":      task_id,
        "score":        score,
        "success":      success,
        "steps_taken":  steps_taken,
        "total_reward": sum(rewards),
    }

# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Legal Contract Review Agent")
    parser.add_argument("--task",  choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--steps", type=int, default=MAX_STEPS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    # OpenAI client pointing at Gemini
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results: List[Dict[str, Any]] = []
    for task_id in tasks:
        result = run_episode(
            client=client,
            task_id=task_id,
            max_steps=args.steps,
            verbose=not args.quiet,
        )
        all_results.append(result)

    # Summary to stderr — keeps stdout clean for the spec parser
    import re
    json_str = json.dumps(all_results, indent=2)
    # Format total_reward and score to strictly have 2 decimal places
    json_str = re.sub(r'"total_reward":\s*(-?\d+(?:\.\d+)?)', lambda m: f'"total_reward": {float(m.group(1)):.2f}', json_str)
    json_str = re.sub(r'"score":\s*(-?\d+(?:\.\d+)?)', lambda m: f'"score": {float(m.group(1)):.2f}', json_str)
    print("\nJSON_RESULTS:", json_str, file=sys.stderr)


if __name__ == "__main__":
    main()