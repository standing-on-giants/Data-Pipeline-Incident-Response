# Context

## Project
Meta RL Hackathon - Round 2

## Current Workspace State
- Data pipeline environment is running for all 3 tasks.
- Inference agent files exist, including inference_gemini.py.
- No prior Instructions.md / Context.md / Change_log.md were present.

## Current Goal
Add round-2 improvements centered on schema drift in the data pipeline scenario.

## Proposed Direction
- Introduce dynamic schema mutation between pipeline runs.
- Add explicit handling logic for schema drift in the agent flow.
- Demonstrate adaptation when fields are renamed (example: spend -> total_spend).
