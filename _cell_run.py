# Dynamically detect which tasks are available in the cloned branch
from src.tasks import TASKS as _AVAILABLE_TASKS

ALL_TASKS = ['easy', 'medium', 'hard', 'hard2']
VALID_TASKS = [t for t in ALL_TASKS if t in _AVAILABLE_TASKS]

print(f'Available tasks in this branch: {VALID_TASKS}', file=sys.stderr)

# Change TASKS_TO_RUN to run a single task for faster testing
# Options: 'easy' | 'medium' | 'hard' | 'hard2' | 'all'
TASKS_TO_RUN = 'all'

task_ids = VALID_TASKS if TASKS_TO_RUN == 'all' else [TASKS_TO_RUN]

all_results = []
for task_id in task_ids:
    if task_id not in _AVAILABLE_TASKS:
        print(f'[SKIP] Task "{task_id}" not available in this branch.', file=sys.stderr)
        continue
    result = run_episode(task_id=task_id, max_steps=MAX_STEPS, verbose=True)
    all_results.append(result)
    torch.cuda.empty_cache()   # release KV cache between tasks

print('\n' + '='*60, file=sys.stderr)
print('FINAL SCORES', file=sys.stderr)
print('='*60, file=sys.stderr)
total = 0.0
for r in all_results:
    status = '[PASSED]' if r['pipeline_passed'] else '[FAILED]'
    print(f"  {r['task_id']:8s} | score={r['score']:.2f} | "
          f"reward={r['total_reward']:+.2f} | steps={r['steps_taken']:2d} | {status}", file=sys.stderr)
    total += r['score']
avg = total / len(all_results) if all_results else 0.0
print(f'\n  Avg score: {avg:.4f}', file=sys.stderr)
print('\nJSON_RESULTS:', json.dumps(all_results, indent=2), file=sys.stderr)
