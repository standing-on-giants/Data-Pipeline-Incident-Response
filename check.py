# openenv_check.py  — paste and run from your repo root
import inspect
from src.environment import DataPipelineEnv
from src.models import PipelineAction, PipelineObservation, StepResult

REQUIRED_METHODS = ["reset", "step", "state", "close"]
REQUIRED_RETURN_TYPES = {
    "reset": PipelineObservation,
    "step": StepResult,
}

print("=== OpenEnv Compliance Check ===\n")

env_methods = [m for m in dir(DataPipelineEnv) if not m.startswith("_")]

for method in REQUIRED_METHODS:
    present = method in env_methods
    print(f"  {'✅' if present else '❌'} DataPipelineEnv.{method}()")

print()

# Type-model checks
for model, name in [(PipelineAction, "PipelineAction"),
                    (PipelineObservation, "PipelineObservation"),
                    (StepResult, "StepResult")]:
    fields = list(model.model_fields.keys())
    print(f"  ✅ {name} — fields: {fields}")

# StepResult must have reward + done
sr_fields = StepResult.model_fields
has_reward = "reward" in sr_fields
has_done   = "done" in sr_fields
print(f"\n  {'✅' if has_reward else '❌'} StepResult.reward present")
print(f"  {'✅' if has_done   else '❌'} StepResult.done present")

# Smoke test
print("\n--- Smoke test ---")
env = DataPipelineEnv(task_id="hard")
obs = env.reset()
assert isinstance(obs, PipelineObservation), "reset() must return PipelineObservation"
action = PipelineAction(action_type="read_data_sample",
                        params={"table": "raw_ads_insights", "n_rows": 5})
result = env.step(action)
assert isinstance(result, StepResult), "step() must return StepResult"
assert hasattr(result, "reward"), "StepResult missing reward"
assert hasattr(result, "done"),   "StepResult missing done"

# The one failing check:
if not hasattr(env, "state") or not callable(getattr(env, "state")):
    print("  ❌ state() missing — add it to DataPipelineEnv (see fix above)")
else:
    s = env.state()
    assert isinstance(s, dict), "state() should return a dict"
    print("  ✅ state() present and callable")

env.close()
print("\nDone.")