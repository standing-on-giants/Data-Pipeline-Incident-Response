## 🎥 2-Minute Demo & Hackathon Pitch
[![Demonstration Video Placeholder](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

### Top-Line Evaluation Results
*(Based on our autonomous 20-episode evaluation benchmark script)*

| Task Difficulty | Baseline (Untrained) | SFT Checkpoint | GRPO Checkpoint |
|-----------------|----------------------|----------------|-----------------|
| Easy | Fails to format actions | *Pending Run* | *Pending Run* |
| Medium | Hallucinates workflows | *Pending Run* | 5/5 Solved (100%) |
| Hard | Gets stuck in loops | *Pending Run* | *Pending Run*|
| Hard2 (Dynamic Drift) | Fails on schema mutations | *Pending Run* | *Pending Run* |

### Why This Matters
Modern data engineering and Platform DevOps teams are bogged down by Level 1 on-call alerts. A "Data Reliability Agent" that can mathematically trace schema drift, autonomously adapt ETL pipelines to upstream API changes, and correctly evaluate when to escalate versus auto-heal is an immensely valuable application of Autonomous AI. By framing real world un-gameable constraints in an OpenEnv environment—and aggressively penalizing "blind-guess" hallucinations through shaped rewards—we lay the groundwork for training agents that can be ultimately trusted with live production data systems. 

### HuggingFace Deployment Repositories
Our pipeline has fully detached the base parameter weights from the PEFT training layers to prevent OOM limits dynamically matching Kaggle hardware constraints. 
*   **SFT LoRA Adapter:** [Abhinav-hf/qwen-sft-lora-adapter](https://huggingface.co/Abhinav-hf/qwen-sft-lora-adapter)
*   **GRPO LoRA Adapter:** [Abhinav-hf/qwen-grpo-lora-adapter](https://huggingface.co/Abhinav-hf/qwen-grpo-lora-adapter)
*   **GRPO (Best Checkpoint):** [Abhinav-hf/qwen-grpo-best-lora-adapter](https://huggingface.co/Abhinav-hf/qwen-grpo-best-lora-adapter)
