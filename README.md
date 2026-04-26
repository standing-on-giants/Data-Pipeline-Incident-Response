---
title: Data Pipeline Incident Qwen GRPO
emoji: 📊
colorFrom: pink
colorTo: pink
sdk: docker
pinned: false
---

# Data Pipeline Incident Response — OpenEnv Environment

**Meta PyTorch OpenEnv Hackathon · Round 2**
**Theme: 3.1 — World Modeling: Professional Tasks**

---

## 🎯 Problem: What capability gap or interesting domain are you targeting?
Data pipelines break silently in production every day. Upstream APIs change payload schemas without warning—columns get renamed, numeric fields arrive as currency strings, deduplication keys shift formats, and join keys grow unexpected prefixes. 

**This environment exists to teach LLMs an anti-hallucination, rigorous diagnostic workflow.** Current LLMs struggle immensely with this domain; when presented with an error, they confidently hallucinate SQL or Python fixes without actually seeking the underlying data context. 

**Is the domain underexplored?** Yes. While standard SWE-bench tasks test writing code for static test suites, few environments address *live incident response* where the world is partially observable, state changes unpredictably (dynamic mid-episode schema drift), and actions carry heavy side-effects. This environment forces the agent to read data samples, compare historical schemas to live ones, and make targeted patches before the pipeline explodes further.

## 🌍 Environment: What does the agent see, do, and get rewarded for?
We place the AI inside a live, broken pipeline DAG (Directed Acyclic Graph) built completely natively with OpenEnv primitives.

- **What the agent sees**: A strict `PipelineObservation` Pydantic model showing upstream failing assertions, the DAG execution history, current and historical schemas, and output data samples (strictly populated only if requested).
- **What the agent does**: Diagnoses via `check_schema`, `read_data_sample`, and `compare_schema`. Fixes the pipeline using structural, surgical patches (`add_data_filter`, `patch_transformation`, `handle_drift`). Escalates unfixable issues natively via `alert_upstream_team`.
- **The Reward Signal (How it actually teaches)**:
  - **Informativeness**: Instead of a 0/1 pass/fail at the end of the episode, the agent receives shaped rewards: `+0.4` for each assertion newly passing after a pipeline run, and `-0.5` for each assertion newly failing.
  - **Hard to Game**: An agent attempting to blindly guess a `patch_transformation` without first reading the data (`read_data_sample` or `check_schema`) receives a strict `-0.5` "blind-fix" penalty. Attempting to sweep errors under the rug (`mark_acceptable` on a failing test) receives a massive `-1.0` penalty. This actively forces methodical diagnosis over naive patching.
  - **Resilience**: In the `hard2` task, dynamic contract schema drift applies dynamically *between* actions. The agent must continuously query the environment, not just memorize a static pre-prompt state.

## 📈 Results: What changed after training?
We collected successful trajectories by interacting with the environment on `easy` and `medium` configurations, creating a dataset for an initial Supervised Fine-Tuning (SFT) phase, followed by a deeply engaged Group Relative Policy Optimization (GRPO) training phase on the dynamic `hard2` tasks utilizing `Qwen2.5-3B-Instruct`.

After training, the GRPO model successfully discarded chaotic zero-shot code-generation reflexes. It learned to sequentially load the schema, check the data, apply safe structural patches, completely avoid the blind-patch penalty, and gracefully handle dynamic contract drift during pipeline validation runs.

![Training Progress Plot — GRPO Reward Stabilization and Improvement Over Training Steps.](media/training_curve.png)

![Evaluation Score Comparison — Trained Qwen GRPO Agent vs. Untrained Baseline across difficulty tasks.](media/eval_results.png)

*For our extensive environment interaction training process and GRPO tuning details, check out `train_grpo/train_grpo_qwen_merged.ipynb` and `inference_qwen_comparison_GRPO_vs_og.py`.*

## 💡 Why does it matter?
Modern data engineering and Platform DevOps teams are bogged down by Level 1 on-call alerts. A "Data Reliability Agent" that can trace schema drift, autonomously adapt ETL pipelines to upstream API changes, and correctly evaluate when to escalate versus auto-heal is an immensely valuable application of Autonomous AI. By framing real world un-gameable constraints in an OpenEnv environment, we lay the groundwork for training agents that can be trusted with live production data systems.

## ⚙️ Clean Engineering (Standard API & OpenEnv Spec)
- Fully respects OpenEnv Client/Server isolation, using strictly the WebSocket `/ws` API standards.
- Implements the strict, predictable Gym-style API `reset()`, `step()`, `state()`.
- Verified and validated `openenv.yaml` specification without any reserved tool name overrides.

### Setup & Run
```bash
# Start the WebSocket server
python -m src.server

# Run Trained Qwen GRPO Evaluation natively
# Uses bitsandbytes 8-bit to fit smoothly on consumer GPUs
conda activate Basic_Computer_vision
python inference_qwen_comparison_GRPO_vs_og.py --use-8bit --models grpo
```

*See `Instructions.md` and `Decisions.md` for extended architectural details and hackathon development logs.*
