"""
train_grpo.py — General SFT -> GRPO Training for Data Pipeline Incident Response
==================================================================================
Model-agnostic CLI script. Works with any Hugging Face causal LM.

Usage:
  python train_grpo.py
  python train_grpo.py --model Qwen/Qwen2.5-3B-Instruct --lora-rank 32
  python train_grpo.py --skip-sft --grpo-only
  python train_grpo.py --model meta-llama/Llama-3.1-8B-Instruct --push-to-hub my-org/my-model
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

import torch
import numpy as np

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.environment import DataPipelineEnv
from src.models import PipelineAction, PipelineObservation

# ------------------------------------------------------------------ #
# System prompt (shared with inference scripts)
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert data engineer diagnosing and fixing broken data pipelines.
You receive pipeline state and must choose ONE action per turn.

WORKFLOW:
1. read_data_sample on raw tables to see the data.
2. check_schema / compare_schema if schema drift is suspected.
3. Apply fix: add_data_filter or patch_transformation.
4. run_pipeline to verify.
5. alert_upstream_team if data is genuinely corrupted.

RULES:
- Respond with ONLY a JSON object. No markdown, no prose.
- Never apply a fix before reading the data.
- Never repeat the same action that didn't work.
- If a "unique" assertion fails, use dedup.
- After parse_currency, always chain coalesce.
""").strip()

# ------------------------------------------------------------------ #
# Gold trajectories for SFT
# ------------------------------------------------------------------ #

GOLD_ACTIONS = {
    "easy": [
        {"action_type": "read_data_sample", "params": {"table": "raw_orders", "n_rows": 20}},
        {"action_type": "add_data_filter", "params": {"step_id": "transform_orders", "filter_condition": "user_id IS NOT NULL"}},
        {"action_type": "run_pipeline", "params": {}},
    ],
    "medium": [
        {"action_type": "read_data_sample", "params": {"table": "raw_order_items", "n_rows": 20}},
        {"action_type": "patch_transformation", "params": {"step_id": "transform_items", "patch_type": "dedup", "column": "order_item_id"}},
        {"action_type": "run_pipeline", "params": {}},
    ],
    "hard": [
        {"action_type": "read_data_sample", "params": {"table": "raw_ads_insights", "n_rows": 20}},
        {"action_type": "compare_schema", "params": {"table": "raw_ads_insights"}},
        {"action_type": "patch_transformation", "params": {"step_id": "transform_insights", "patch_type": "parse_currency", "column": "spend"}},
        {"action_type": "patch_transformation", "params": {"step_id": "transform_insights", "patch_type": "coalesce", "column": "spend"}},
        {"action_type": "patch_transformation", "params": {"step_id": "transform_insights", "patch_type": "parse_currency", "column": "impressions"}},
        {"action_type": "add_data_filter", "params": {"step_id": "transform_insights", "filter_condition": "impressions IS NOT NULL"}},
        {"action_type": "run_pipeline", "params": {}},
        {"action_type": "patch_transformation", "params": {"step_id": "transform_conversions", "patch_type": "dedup", "column": "event_id"}},
        {"action_type": "run_pipeline", "params": {}},
        {"action_type": "compare_schema", "params": {"table": "raw_conversions"}},
        {"action_type": "patch_transformation", "params": {"step_id": "transform_conversions", "patch_type": "strip_prefix", "column": "campaign_id"}},
        {"action_type": "patch_transformation", "params": {"step_id": "transform_conversions", "patch_type": "cast_column", "column": "campaign_id"}},
        {"action_type": "run_pipeline", "params": {}},
        {"action_type": "alert_upstream_team", "params": {"team": "meta_ads_api_team", "issue_description": "Graph API outage: N/A impressions"}},
    ],
}


def format_obs_for_training(obs: PipelineObservation, step: int) -> str:
    """Convert observation to the text prompt shown to the model."""
    failed = "\n".join(
        f"  [{r.assertion_id}] {r.assertion_type} on {r.table}({r.column or 'N/A'}): {r.actual}"
        for r in obs.failed_assertions
    ) or "  (none)"
    passed = ", ".join(r.assertion_id for r in obs.passed_assertions) or "none"
    dag = "\n".join(
        f"  {n.step_id}: {n.input_table} -> {n.output_table}"
        + (f" | filters: {n.applied_filters}" if n.applied_filters else "")
        + (f" | patches: {n.applied_patches}" if n.applied_patches else "")
        for n in obs.dag_structure
    )
    hist = "\n".join(f"  {r.date}: {r.status} ({r.row_count} rows)" for r in obs.historical_runs)
    schema = ""
    if obs.current_schema:
        schema += "\nCURRENT SCHEMA: " + json.dumps(obs.current_schema)
    if obs.schema_diff:
        schema += "\nSCHEMA DIFF: " + json.dumps(obs.schema_diff)
    sample = ""
    if obs.data_sample:
        sample = "\nDATA SAMPLE: " + json.dumps(obs.data_sample[:3], default=str)
    actions = "\n".join(f"  {a}" for a in obs.actions_taken[-4:]) or "  (none)"

    return textwrap.dedent(f"""
    STEP {step}/{obs.max_steps} | TASK: {obs.task_id} ({obs.difficulty})
    DESCRIPTION: {obs.description}
    PIPELINE PASSED: {obs.pipeline_passed}
    LAST ACTION RESULT: {obs.last_action_result}
    DAG:\n{dag}
    FAILING:\n{failed}
    PASSING: {passed}
    HISTORY:\n{hist}
    RECENT ACTIONS:\n{actions}
    {sample}{schema}
    Respond with exactly ONE action JSON object.
    """).strip()


def collect_gold_trajectories(task_ids: List[str], n_episodes: int = 10) -> List[tuple]:
    """Run env with gold actions to collect (obs_text, action_json) pairs for SFT."""
    pairs = []
    for task_id in task_ids:
        gold = GOLD_ACTIONS.get(task_id, [])
        if not gold:
            continue
        for _ in range(n_episodes):
            env = DataPipelineEnv(task_id=task_id)
            obs = env.reset()
            for step_idx, action_dict in enumerate(gold, 1):
                obs_text = format_obs_for_training(obs, step_idx)
                action = PipelineAction(**action_dict)
                pairs.append((obs_text, json.dumps(action_dict)))
                result = env.step(action)
                obs = result.observation
                if obs.pipeline_passed:
                    break
    return pairs


# ------------------------------------------------------------------ #
# Reward function for GRPO
# ------------------------------------------------------------------ #

def parse_action_from_text(text: str) -> Optional[PipelineAction]:
    """Parse a PipelineAction from model output text."""
    import re
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.DOTALL).strip()
    if "```" in text:
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}") + 1
    if end <= start:
        return None
    try:
        data = json.loads(text[start:end])
        if "action_type" in data:
            return PipelineAction(**data)
    except Exception:
        pass
    return None


def pipeline_reward_fn(completions, prompts=None, **kwargs) -> list:
    """
    GRPO reward function. For each completion:
    - Parse action JSON -> execute in env -> return shaped reward.
    - format_bonus (+0.3) for valid JSON
    - drift_bonus (+0.3) for detecting real schema drift
    - malformed_penalty (-0.3) for unparseable output
    """
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else completion[0].get("content", "")
        action = parse_action_from_text(text)

        if action is None:
            rewards.append(-0.3)
            continue

        # Valid JSON bonus
        reward = 0.3

        # Execute in a fresh environment to get step reward
        try:
            env = DataPipelineEnv(task_id="hard")
            obs = env.reset()
            result = env.step(action)
            reward += result.reward or 0.0

            # Drift detection bonus
            if action.action_type == "compare_schema":
                if result.observation.schema_diff and len(result.observation.schema_diff) > 0:
                    reward += 0.3
        except Exception:
            reward -= 0.2

        rewards.append(float(reward))

    return rewards


# ------------------------------------------------------------------ #
# Evaluation
# ------------------------------------------------------------------ #

def run_eval_episode(model, tokenizer, task_id: str, max_steps: int = 20) -> dict:
    """Run a single evaluation episode and return score metrics."""
    env = DataPipelineEnv(task_id=task_id)
    obs = env.reset()
    rewards = []
    step = 0

    for step in range(1, max_steps + 1):
        if obs.pipeline_passed:
            break

        prompt_text = format_obs_for_training(obs, step)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            out_ids = model.generate(
                inputs, max_new_tokens=200, temperature=0.1,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(out_ids[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        action = parse_action_from_text(response)
        if action is None:
            action = PipelineAction(action_type="compare_schema", params={"table": "raw_orders"})

        result = env.step(action)
        obs = result.observation
        rewards.append(result.reward or 0.0)
        if result.done:
            break

    n_total = len(obs.failed_assertions) + len(obs.passed_assertions)
    n_passed = len(obs.passed_assertions)
    score = min(max(n_passed / n_total if n_total > 0 else 0, 0.01), 0.99)

    return {
        "task_id": task_id,
        "score": round(score, 3),
        "pipeline_passed": obs.pipeline_passed,
        "total_reward": round(sum(rewards), 3),
        "steps": step,
    }


# ------------------------------------------------------------------ #
# Main training pipeline
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="SFT -> GRPO training for Data Pipeline Agent")
    parser.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct", help="HF model name")
    parser.add_argument("--sft-epochs", type=int, default=3)
    parser.add_argument("--grpo-epochs", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--kl-coeff", type=float, default=0.1)
    parser.add_argument("--num-generations", type=int, default=4, help="G: completions per prompt")
    parser.add_argument("--sft-tasks", default="easy,medium", help="Comma-separated task IDs for SFT")
    parser.add_argument("--grpo-tasks", default="hard,hard2", help="Comma-separated task IDs for GRPO")
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--push-to-hub", default=None, help="HF repo name to push trained model")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN", "")
    sft_tasks = [t.strip() for t in args.sft_tasks.split(",")]
    grpo_tasks = [t.strip() for t in args.grpo_tasks.split(",")]

    print(f"Model:           {args.model}")
    print(f"LoRA rank:       {args.lora_rank}")
    print(f"SFT tasks:       {sft_tasks}")
    print(f"GRPO tasks:      {grpo_tasks}")
    print(f"KL coefficient:  {args.kl_coeff}")
    print(f"G (completions): {args.num_generations}")
    print(f"Output dir:      {args.output_dir}")

    # ── 1. Load model ──────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        token=hf_token,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_rank,
        lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({100*trainable/total:.2f}%)")

    # ── 2. SFT ─────────────────────────────────────────────────────
    sft_dir = os.path.join(args.output_dir, "sft")
    if not args.skip_sft:
        print(f"\n[2/5] Collecting SFT trajectories from {sft_tasks}...")
        gold_pairs = collect_gold_trajectories(sft_tasks, n_episodes=10)
        print(f"  Collected {len(gold_pairs)} (observation, action) pairs")

        sft_texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": obs_text},
                 {"role": "assistant", "content": action_json}],
                tokenize=False, add_generation_prompt=False,
            )
            for obs_text, action_json in gold_pairs
        ]

        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import is_bfloat16_supported

        sft_dataset = Dataset.from_dict({"text": sft_texts})
        print(f"  SFT dataset: {len(sft_dataset)} examples")

        sft_trainer = SFTTrainer(
            model=model, tokenizer=tokenizer,
            train_dataset=sft_dataset,
            dataset_text_field="text",
            max_seq_length=4096,
            args=TrainingArguments(
                report_to="none",
                average_tokens_across_devices=False,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                num_train_epochs=args.sft_epochs,
                warmup_ratio=0.1,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                output_dir=sft_dir,
                save_steps=50,
                seed=42,
            ),
        )
        print("\n[3/5] Training SFT...")
        sft_stats = sft_trainer.train()
        print(f"  SFT complete. Loss: {sft_stats.training_loss:.4f}")
        model.save_pretrained(sft_dir)
        tokenizer.save_pretrained(sft_dir)
    else:
        print("\n[2/5] Skipping SFT (--skip-sft)")
        print("[3/5] Skipping SFT training")

    # ── 3. GRPO ────────────────────────────────────────────────────
    print(f"\n[4/5] Building GRPO prompts from {grpo_tasks}...")

    # Filter to only available tasks
    from src.tasks import TASKS as available_tasks
    valid_grpo = [t for t in grpo_tasks if t in available_tasks]
    if not valid_grpo:
        print(f"  WARNING: No valid GRPO tasks found. Available: {list(available_tasks.keys())}")
        valid_grpo = ["hard"]

    grpo_prompts = []
    for task_id in valid_grpo:
        for _ in range(25):
            env = DataPipelineEnv(task_id=task_id)
            obs = env.reset()
            prompt_text = format_obs_for_training(obs, step=1)
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True,
            )
            grpo_prompts.append({"prompt": chat_prompt})

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import is_bfloat16_supported

    grpo_dataset = Dataset.from_list(grpo_prompts)
    print(f"  GRPO dataset: {len(grpo_dataset)} prompts")

    grpo_dir = os.path.join(args.output_dir, "grpo")
    grpo_config = GRPOConfig(
        output_dir=grpo_dir,
        num_generations=args.num_generations,
        max_completion_length=200,
        temperature=0.8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=args.grpo_epochs,
        learning_rate=5e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        beta=args.kl_coeff,
        loss_type="grpo",
        logging_steps=1,
        save_steps=50,
        seed=42,
        max_prompt_length=4096,
        max_grad_norm=1.0,
        warmup_steps=1,
    )

    grpo_trainer = GRPOTrainer(
        model=model, tokenizer=tokenizer,
        reward_funcs=pipeline_reward_fn,
        args=grpo_config,
        train_dataset=grpo_dataset,
    )

    print(f"  KL coeff: {args.kl_coeff}, G: {args.num_generations}")
    print("\n  Starting GRPO training...")
    grpo_stats = grpo_trainer.train()
    print(f"  GRPO complete. Loss: {grpo_stats.training_loss:.4f}")
    model.save_pretrained(grpo_dir)
    tokenizer.save_pretrained(grpo_dir)

    # ── 4. Evaluate ────────────────────────────────────────────────
    print("\n[5/5] Evaluating trained model...")
    from unsloth import FastLanguageModel as FLM
    FLM.for_inference(model)

    eval_tasks = [t for t in ["easy", "medium", "hard", "hard2"] if t in available_tasks]
    baseline = {"easy": 0.70, "medium": 0.55, "hard": 0.30, "hard2": 0.30}

    print(f"\n{'Task':<10} {'Baseline':<10} {'Trained':<10} {'Delta':<10} {'Pass?'}")
    print("=" * 50)
    for task_id in eval_tasks:
        r = run_eval_episode(model, tokenizer, task_id=task_id)
        base = baseline.get(task_id, 0.0)
        delta = r["score"] - base
        status = "[PASSED]" if r["pipeline_passed"] else "[FAILED]"
        sign = "+" if delta >= 0 else ""
        print(f"{task_id:<10} {base:<10.3f} {r['score']:<10.3f} {sign}{delta:<9.3f} {status}")

    # ── 5. Push to Hub ─────────────────────────────────────────────
    if args.push_to_hub and hf_token:
        print(f"\nPushing to HF Hub: {args.push_to_hub}")
        model.push_to_hub_merged(
            args.push_to_hub, tokenizer,
            save_method="merged_16bit", token=hf_token,
        )
        print(f"Model pushed to https://huggingface.co/{args.push_to_hub}")

    print("\nDone!")


if __name__ == "__main__":
    main()
