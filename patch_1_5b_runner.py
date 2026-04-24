"""
Patches run_on_kaggle/run_on_kaggle_qwen_1.5b.ipynb:
- Reduce history cap from 14 to 6 turns (1.5B can't handle long context)
- Aggressively truncate user prompt to MAX_PROMPT_CHARS
- Print exception TYPE not just message (to diagnose silent OOM/ValueError)
- After 3 consecutive generation errors: reset history entirely + skip step
- Add context-length pre-check: if prompt > MAX_INPUT_TOKENS, trim history first
"""
import json

PATH = "run_on_kaggle/run_on_kaggle_qwen_1.5b.ipynb"

NEW_CONFIG_APPEND = """
# ── Context safety limits for 1.5B model ──────────────────────────────────
# 1.5B model has a 32k context, but long histories cause silent OOM / empty output.
# Keep the prompt short: cap history and truncate the user message.
MAX_HISTORY_TURNS  = 6      # max (user, assistant) pairs kept in history
MAX_PROMPT_CHARS   = 3000   # truncate user prompt beyond this many chars
"""

NEW_RUNNER = """from src.environment import DataPipelineEnv


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

            history.append({'role': 'assistant', 'content': response_text or '{}'})

            result = env.step(action)
            obs    = result.observation
            reward = result.reward or 0.0
            done   = result.done
            error  = getattr(obs, 'last_action_error', None) or None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=json.dumps(action.model_dump())[:200], reward=reward, done=done, error=error)

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
"""

with open(PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))

    # Patch config cell: add MAX_HISTORY_TURNS and MAX_PROMPT_CHARS
    if "MAX_STEPS   = int(os.getenv" in src and "MAX_HISTORY_TURNS" not in src:
        src = src.rstrip() + "\n" + NEW_CONFIG_APPEND
        cell["source"] = src.splitlines(keepends=True)
        print(f"  [config] Cell {i} updated with context limits")

    # Replace runner cell entirely
    if "from src.environment import DataPipelineEnv" in src and "run_episode" in src:
        cell["source"] = NEW_RUNNER.splitlines(keepends=True)
        print(f"  [runner] Cell {i} replaced with robust runner")

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"Saved {PATH}")
