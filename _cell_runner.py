from src.environment import DataPipelineEnv


def log_start(task: str, env: str, model: str) -> None:
    print(f'[START] task={task} env={env} model={model}', flush=True)

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


def run_episode(task_id: str, max_steps: int = MAX_STEPS, verbose: bool = True) -> dict:
    env = DataPipelineEnv(task_id=task_id)

    history:         List[dict] = []
    rewards:         List[float] = []
    steps_taken:     int = 0
    score:           float = 0.0
    success:         bool = False
    n_passed:        int = 0
    n_total:         int = 0
    pipeline_passed: bool = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        if verbose:
            print(f'\n{"="*60}', file=sys.stderr)
            print(f'TASK: {task_id.upper()}  |  {len(obs.failed_assertions)} assertions failing', file=sys.stderr)
            print(f'{"="*60}', file=sys.stderr)
            print(f'Description: {obs.description}', file=sys.stderr)

        for step in range(1, max_steps + 1):
            if obs.pipeline_passed:
                if verbose:
                    print(f'\n[PASSED] Pipeline passed at step {step - 1}!', file=sys.stderr)
                break

            # If the model is in a loop, trim history aggressively to break the pattern
            looped = _detect_action_loop(obs.actions_taken)
            if looped:
                if verbose:
                    print(f'  [Step {step}] Loop detected: {looped}. Trimming history.', file=sys.stderr)
                # Keep only last 2 turns so the model loses the repetitive context
                history = history[-2:]

            user_prompt = build_user_prompt(obs, step)
            history.append({'role': 'user', 'content': user_prompt})
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + history

            torch.cuda.empty_cache()

            response_text = ''
            try:
                response_text = generate(messages)
            except torch.cuda.OutOfMemoryError:
                print(f'[OOM] Step {step}: trimming history to 4 turns and retrying.', file=sys.stderr)
                history = history[-4:]
                messages = [{'role': 'system', 'content': SYSTEM_PROMPT}] + history
                torch.cuda.empty_cache()
                try:
                    response_text = generate(messages)
                except Exception as e2:
                    print(f'[OOM-RETRY-FAIL] {e2}', file=sys.stderr)
            except Exception as exc:
                if verbose:
                    print(f'  [Step {step}] Generation error: {exc}. Using fallback.', file=sys.stderr, flush=True)

            action = parse_llm_response(response_text)

            # Smart fallback: empty response -> diagnostic instead of run_pipeline
            if action.action_type == 'run_pipeline' and not response_text.strip():
                if obs.failed_assertions:
                    target_table = obs.failed_assertions[0].table
                    action = PipelineAction(
                        action_type='compare_schema',
                        params={'table': target_table}
                    )

            history.append({'role': 'assistant', 'content': response_text or '{}'})
            if len(history) > 14:
                history = history[-14:]

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
                print(f'\n[Step {step}] Raw response: {response_text[:120]}', file=sys.stderr)
                print(f'[Step {step}] Action: {action.action_type}({action.params})', file=sys.stderr)
                print(
                    f'  Reward: {reward:+.2f} | '
                    f'Passed: {len(obs.passed_assertions)}/{len(obs.failed_assertions)+len(obs.passed_assertions)} | '
                    f'Result: {obs.last_action_result[:80]}',
                    file=sys.stderr
                )

            if done:
                break

        n_total  = len(obs.failed_assertions) + len(obs.passed_assertions)
        n_passed = len(obs.passed_assertions)
        pipeline_passed = obs.pipeline_passed
        raw_score = n_passed / n_total if n_total > 0 else 0.0
        score   = min(max(raw_score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

        if verbose:
            print(f'\n--- Episode Summary ---', file=sys.stderr)
            print(f'  Score (assertion pass rate): {score:.2f}', file=sys.stderr)
            print(f'  Total reward:                {sum(rewards):.2f}', file=sys.stderr)
            print(f'  Steps taken:                 {steps_taken}', file=sys.stderr)
            print(f'  Pipeline passed:             {pipeline_passed}', file=sys.stderr)

    except Exception as exc:
        import traceback
        print(f'[ERROR] {task_id}: {exc}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    finally:
        try:
            env.close()
        except AttributeError:
            pass
        except Exception as e:
            print(f'[DEBUG] env.close() error: {e}', file=sys.stderr, flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        'task_id':           task_id,
        'score':             round(score, 4),
        'success':           success,
        'pipeline_passed':  pipeline_passed,
        'total_reward':     round(sum(rewards), 4),
        'steps_taken':      steps_taken,
        'assertions_passed': n_passed,
        'assertions_total':  n_total,
    }


print('Runner ready.')
