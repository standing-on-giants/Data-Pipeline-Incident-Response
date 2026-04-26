# Video Script (2-Minute Hackathon Pitch)

**Total Time:** 120 seconds
**Pacing:** ~130-150 words per minute
**Theme Alignment:** Professional Tasks & Long-Horizon Planning (Theme 3.1 & 2)

---

### [0:00–0:15] THE CAPABILITY GAP (Storytelling Hook)
**NARRATION:**  
"When a data pipeline breaks at 3 AM, an engineer reads logs, inspects schemas, applies patches, and reruns the pipeline until the board is green. It's fully reactive. We built the OpenEnv 'Data Pipeline Incident Response' environment to train LLMs to do exactly that autonomously."

**ON-SCREEN:**  
Fast-paced split screen. On the left: a frantic Slack stream of "Failing Assertion" alerts waking up an engineer. On the right: our agent terminal receiving the exact same structured alert payload and starting its thought process.

### [0:15–0:55] ENVIRONMENT INNOVATION (40%)
**NARRATION:**  
"What makes this environment novel is strict partial observability. The agent doesn't see ground truth. It has to navigate multi-step diagnoses using 11 distinct tools—including drift detection, currency parsing, and schema rename resolution—across 4 difficulty tiers mapping real production failure modes. 

But here is the twist: In our hardest tasks, we introduce dynamic schema drift mid-episode. As the agent applies fixes, an upstream API silently mutates its payload format. The world shifts while the agent is fixing it, forcing the model to continuously re-check schemas and adapt its policy on the fly to survive."

**ON-SCREEN:**  
Clean screen recording of the environment interface architecture. Callouts zooming into the 11-action table (highlighting `read_data_sample`, `compare_schema`, `handle_drift`). A DAG visualization of the pipeline. A flashing graphic showing "Dynamic Drift Event: 'spend' renamed to 'total_spend'" interrupting the agent's run.

### [0:55–1:15] REWARD & PIPELINE (10%)
**NARRATION:**  
"To teach this professional incident hygiene, we trained the Qwen2.5 3B model over a custom Group Relative Policy Optimization pipeline. 

Our environment uses a sparse, un-gameable shaped reward. If the agent blindly guesses a patch without first reading the data, it gets a massive penalty. It must read, it must diagnose, and only then is it rewarded for pushing passing assertions."

**ON-SCREEN:**  
Syntax-highlighted code snippet of the `pipeline_reward_fn()` focusing on the strict `-0.5` blind-fix penalty. Quick transition to the GRPO training reward curve PNG, showing the steep learning curve overcoming early chaotic penalties.

### [1:15–1:45] SHOWING IMPROVEMENT (20%)
**NARRATION:**  
"The results were transformative. Before training, the model acted like a chaotic code-generator. It would blindly hallucinate actions that didn't even exist—like 'alert_owner'—without ever actually looking at the data. 

After GRPO training, the model executes a rigid, precise incident sequence: read data sample, compare the schema, patch the deduplication key, and rerun the pipeline safely. With this, we achieved a 100% solve rate on our medium pipeline tasks."

**ON-SCREEN:**  
Side-by-side terminal log comparison. 
*Left (Before):* Model outputs invalid JSON, tries `action: alert_owner`, receives error penalty.
*Right (After):* Tidy sequential green execution: `read_data_sample` -> `compare_schema` -> `patch_transformation(dedup)` -> `run_pipeline` —> `[PASSED] ALL ASSERTIONS GREEN`.

### [1:45–2:00] THE FUTURE (Storytelling Resolution)
**NARRATION:**  
"While the model mastered isolated components, the easiest and hardest tasks didn't transfer perfectly at this step scale. This honest outcome gives us a clear signal: broader SFT data trajectories and extended GRPO compute will close that gap. We've proven the foundation for the fully autonomous, Level-1 data engineer."

**ON-SCREEN:**  
Final evaluation rankings table (SFT vs. GRPO metrics). Fade out to the team roster and HuggingFace repo link.
