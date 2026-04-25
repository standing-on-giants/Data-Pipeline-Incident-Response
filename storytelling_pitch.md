# Storytelling & Presentation (Hackathon Pitch)

**Criterion:** Storytelling & Presentation
**Weight:** 30%

Here are the direct answers you can use for your presentation, demo video, or submission form:

### 1. What is the problem?
Data pipelines break every single day in production. A vendor silently changes an API export format, numeric fields arrive as currency strings, deduplication keys shift, or join keys grow unexpected prefixes. Today, Data Engineers are woken up at 2 AM to manually write SQL queries, check schemas, and push hotfixes. It’s tedious, manual, and reactive. We need an AI agent that can act as an autonomous Level 1 On-Call Data Engineer.

### 2. What is the environment?
We built **Data Pipeline Incident Response — OpenEnv**, a fully interactive incident room for AI agents. Instead of just "generating code," the agent is dropped into a live, broken pipeline graph. It receives an alert (failing data quality assertions) and must iteratively query the data (`read_data_sample`, `compare_schema`), diagnose the root cause, and apply structural patches (`add_data_filter`, `patch_transformation`). 

**The twist:** We injected *dynamic, mid-episode schema drift*. As the agent applies fixes, the upstream API mutates again in real-time, forcing the agent to detect and adapt to live contract changes rather than just solving a static puzzle.

### 3. What did the agent learn? (The GRPO / Qwen Journey)
At first, smaller models like Qwen 1.5B/3B and LLaMA 3.1 8B failed completely. They would hallucinate fixes without looking at the data, get stuck in infinite loops of repeating the same action, or blindly sweep errors under the rug.

To fix this, we used **Group Relative Policy Optimization (GRPO)** and aggressive reward shaping. We penalized the agent severely for "blind fixes" (-0.5 reward) or repeating the same action without progress. We rewarded it for correctly diagnosing drift and passing the pipeline. 

**The result:** The agent learned a disciplined, professional workflow. It learned that it *must* inspect the schema before patching. It learned to break out of its own loops by reviewing its trajectory history. Ultimately, it transformed from a chaotic code-generator into a methodical, reasoning-driven Data Engineer.

### 4. Is the demo engaging and easy to follow for a non-technical audience?
**How to present this in your demo:**
*   **The Hook:** Start with a relatable scenario: "It's 2 AM. The finance dashboard is broken because Meta changed their Ad API payload again."
*   **The Action:** Show the agent waking up. Show the terminal output where the agent *reads the data*, sees the unexpected `"CMP_"` prefix, and applies a `strip_prefix` patch.
*   **The Twist:** Show the pipeline failing *again* because the API drifted mid-run. Watch the agent dynamically react, call `compare_schema`, notice the column rename, and fix it on the fly.
*   **The Resolution:** The pipeline passes. All assertions green. The dashboard is saved. 
*   **The Visuals:** Keep the focus on the agent's thought process (the trajectory log) rather than the underlying Python code. The logs read like a detective story.
