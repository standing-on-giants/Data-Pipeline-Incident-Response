# From 2 AM Pipeline Failures to Adaptive RL Agents

Meta OpenEnv Hackathon India 2026 - Round 2 submission by Team [Your Team Name]

## TL;DR

We built an OpenEnv-compliant environment where an LLM acts like an on-call data engineer. The agent receives failing pipeline alerts, investigates the cause, applies targeted fixes, reruns the pipeline, and handles live schema drift injected mid-episode.

This project is primarily aligned with Theme 3.1 (World Modeling: Professional Tasks), with a secondary alignment to Theme 2 (Long-Horizon Planning), because the world changes while the agent is already fixing it.

## The problem we chose

Real data pipelines fail constantly in production:

- Columns get renamed without notice
- Numeric fields arrive as strings like "$1,234" or "N/A"
- Join keys silently change format
- Duplicate events appear from API retries

Most teams still solve this manually in incident mode. We wanted an agent that can follow a disciplined incident workflow instead of guessing.

## The environment we built

Environment name: Data Pipeline Incident Response (OpenEnv)

The agent operates over a DAG-style pipeline and receives:

- Failing/passing data quality assertions
- Historical run info
- Current vs historical schema (after inspection actions)
- Step-by-step action history

### Action space (11 typed actions)

Examples:

- read_data_sample
- check_schema
- compare_schema
- handle_drift
- add_data_filter
- patch_transformation
- run_pipeline
- alert_upstream_team

### Why this is not a toy benchmark

The environment is partially observable and tool-driven. The model cannot pass by pure text pattern matching. It must call the right diagnostic tools, patch the right transformation step, then validate by rerunning the pipeline.

## What makes Round 2 novel in our setup

The hard2 task includes dynamic drift scheduled during run_pipeline:

1. spend -> total_spend rename
2. auth format rotation
3. tighter rate limit behavior

So even if the model starts with a correct plan, the world shifts. The agent must re-check schemas and adapt policy mid-trajectory.

This is exactly the kind of persistent world-model behavior we wanted to train.

## Theme mapping

Primary:

- Theme 3.1 - World Modeling (Professional Tasks)

Secondary:

- Theme 2 - Long-Horizon Planning and Instruction Following

Reason:

- Professional because it mirrors real incident response in data engineering
- Long-horizon because delayed reward and sequential dependencies matter (diagnose -> patch -> rerun -> re-diagnose)

## Training pipeline

We used a two-stage pipeline:

1. SFT stage

- Collect successful trajectories on easier tasks
- Teach format discipline and basic diagnostic behavior

2. GRPO stage

- Train on hard/hard2 with shaped rewards
- Encourage behaviors like schema inspection before patching and adaptive handling of drift

Training stack:

- Unsloth + HF TRL (GRPO)
- OpenEnv environment loop for reward generation

## Reward design (what we optimized for)

Core rewards include:

- Positive reward when failing assertions become passing after run_pipeline
- Penalty for regressions
- Reward for correct escalation only when required
- Penalty for blind fixes and non-progress loops

Design principle:
Reward should reflect professional incident hygiene, not just terminal pass/fail.

## Observable improvement (what we will show judges)

We evaluate baseline vs trained policy on easy/medium/hard/hard2:

- Per-task scores
- Average score
- Before/after trajectories on hard2

In our runs, untrained models tend to:

- Guess patches without reading data
- Loop on repeated actions
- Miss post-drift rename handling

After training, policies become more procedural:

- Inspect first
- Patch with better ordering
- Re-validate and adapt when drift appears

## Why this matters beyond the hackathon

This maps directly to enterprise needs:

- Faster incident triage
- Lower analyst/on-call load
- Better guardrails against silent data corruption

The same design pattern can extend to:

- ETL reliability assistants
- Monitoring-triggered remediation agents
- Data contract change management workflows

## OpenEnv compliance and artifacts

Minimum requirements covered:

- OpenEnv-compatible environment and server
- Minimal training script/notebook using Unsloth/HF TRL
- This mini-blog for the submission
- Environment intended to be hosted on Hugging Face Spaces

## Quick demo narrative (for a <2 min video)

1. Start at failing assertions on hard2
2. Show agent reading sample/schema before patching
3. Show first repair and run_pipeline
4. Show new drift event appears
5. Show compare_schema + handle_drift usage
6. End with improved assertion pass count and final report

## Links (replace before publishing)

- Environment repo: [Add GitHub/HF link]
- HF Space URL: [Add Space URL]
- Training notebook: [Add notebook link]
- Demo video (optional): [Add YouTube link]

## Team

- [Name 1]
- [Name 2]
- [Name 3]

If you are building LLM agents for real operations, we'd love to collaborate.
