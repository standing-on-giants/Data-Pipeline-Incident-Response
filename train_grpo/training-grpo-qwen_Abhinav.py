import os
import sys
import json
import textwrap
import random
import re
import subprocess
from pathlib import Path

# ==============================================================================
# 1. SETUP & INSTALLATION
# ==============================================================================
print("--- Starting Setup & Installation ---")

# Execute installations via subprocess to ensure this works universally 
# whether run as a pure .py file or pasted into a Jupyter Notebook cell.
subprocess.run(
    "pip install -qU --no-cache-dir unsloth unsloth_zoo transformers trl peft accelerate "
    "bitsandbytes datasets pandas openai python-dotenv", 
    shell=True, check=True
)

# Clearing Unsloth cached modules. This resolves the notorious CausalMask bug 
# where Unsloth fails to load previously compiled Triton kernels after a transformers update.
subprocess.run("rm -rf /kaggle/working/unsloth_compiled_cache", shell=True)
subprocess.run("rm -rf /kaggle/working/Meta_hackathon/unsloth_compiled_cache", shell=True)
print('Installation and Cache Clearance complete.')

# ==============================================================================
# 2. AUTHENTICATION & REPOSITORY CLONING
# ==============================================================================
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    GITHUB_TOKEN = _s.get_secret('GITHUB_TOKEN_DEVARMANI')
    HF_TOKEN = _s.get_secret('hugging_face_access_token') or _s.get_secret('HF_TOKEN')
except Exception:
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')
    HF_TOKEN = os.getenv('HF_TOKEN', '')

REPO_DIR = '/kaggle/working/Meta_hackathon'

if not os.path.exists(REPO_DIR):
    subprocess.run(f"git clone -b dev/pratham https://{GITHUB_TOKEN}@github.com/standing-on-giants/Meta_hackathon.git {REPO_DIR}", shell=True)
else:
    os.chdir(REPO_DIR)
    subprocess.run("git fetch origin && git checkout dev/pratham && git pull origin dev/pratham", shell=True)

# Changing directory using OS module instead of %cd (which is notebook-only)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print(f'Working directory set to: {os.getcwd()}')

import torch
import numpy as np

# Import environment pipeline structures
from src.environment import DataPipelineEnv
from src.models import PipelineAction, PipelineObservation

print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

# ==============================================================================
# 3. LOADING THE MODEL (8-BIT QUANTIZATION)
# ==============================================================================
from unsloth import FastLanguageModel

# We forcefully limit the context window to 2048 to prevent Kaggle T4 OOMs
MAX_SEQ_LENGTH = 2048
MODEL_NAME = 'Qwen/Qwen2.5-3B-Instruct'

print(f"--- Loading Base Model {MODEL_NAME} ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,             # Auto-selects based on hardware capabilities
    load_in_4bit=False,     # Disabled as requested
    load_in_8bit=True,      # Enabled 8-bit precision to fit gradients smoothly in 16GB VRAM
    token=HF_TOKEN,
)
print(f'Model loaded: {MODEL_NAME}')
print(f'Parameters: {model.num_parameters()/1e9:.2f}B')

# Attaching LoRA Adapters (Rank 32 provides high representation capability since base handles 8-bit)
model = FastLanguageModel.get_peft_model(
    model, r=32,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_alpha=32, lora_dropout=0.0, bias='none',
    use_gradient_checkpointing='unsloth', random_state=42,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Trainable Parameters: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({100*trainable/total:.2f}%)')

# ==============================================================================
# 4. SUPERVISED FINE-TUNING (SFT) - STAGE 1
# ==============================================================================
# The model warms up on these deterministic, successful trajectory paths.
# We explicitly supply 'hard' and 'hard2' task behaviors to anchor the schema drift rules.
SYSTEM_PROMPT = textwrap.dedent('''
You are an expert data engineer diagnosing and fixing broken data pipelines.
You receive pipeline state and must choose ONE action per turn.
WORKFLOW: 1. read_data_sample 2. check_schema/compare_schema 3. Apply fix 4. run_pipeline
RULES: Respond with ONLY a JSON object. Never repeat failing actions. dedup for uniqueness failures.
''').strip()

GOLD_ACTIONS = {
    'easy': [
        {'action_type': 'read_data_sample', 'params': {'table': 'raw_orders', 'n_rows': 20}},
        {'action_type': 'add_data_filter', 'params': {'step_id': 'transform_orders', 'filter_condition': 'user_id IS NOT NULL'}},
        {'action_type': 'run_pipeline', 'params': {}},
    ],
    'medium': [
        {'action_type': 'read_data_sample', 'params': {'table': 'raw_order_items', 'n_rows': 20}},
        {'action_type': 'patch_transformation', 'params': {'step_id': 'transform_items', 'patch_type': 'dedup', 'column': 'order_item_id'}},
        {'action_type': 'run_pipeline', 'params': {}},
    ],
    'hard': [
        {'action_type': 'read_data_sample', 'params': {'table': 'raw_ads_insights', 'n_rows': 20}},
        {'action_type': 'compare_schema', 'params': {'table': 'raw_ads_insights'}},
        {'action_type': 'patch_transformation', 'params': {'step_id': 'transform_ads', 'patch_type': 'parse_currency', 'column': 'spend'}},
        {'action_type': 'patch_transformation', 'params': {'step_id': 'transform_ads', 'patch_type': 'coalesce', 'column': 'spend'}},
        {'action_type': 'run_pipeline', 'params': {}},
    ],
    'hard2': [
        {'action_type': 'read_data_sample', 'params': {'table': 'raw_campaigns', 'n_rows': 20}},
        {'action_type': 'compare_schema', 'params': {'table': 'raw_campaigns'}},
        {'action_type': 'patch_transformation', 'params': {'step_id': 'join_campaigns', 'patch_type': 'strip_prefix', 'column': 'campaign_id'}},
        {'action_type': 'patch_transformation', 'params': {'step_id': 'join_campaigns', 'patch_type': 'cast_column', 'column': 'campaign_id'}},
        {'action_type': 'run_pipeline', 'params': {}},
    ],
}

def format_obs(obs, step):
    failed = '\n'.join(f'  [{r.assertion_id}] {r.assertion_type} on {r.table}({r.column or "N/A"}): {r.actual}' for r in obs.failed_assertions) or '  (none)'
    passed = ', '.join(r.assertion_id for r in obs.passed_assertions) or 'none'
    dag = '\n'.join(f'  {n.step_id}: {n.input_table} -> {n.output_table}' for n in obs.dag_structure)
    hist = '\n'.join(f'  {r.date}: {r.status} ({r.row_count} rows)' for r in obs.historical_runs)
    schema = ''
    if obs.current_schema: schema += '\nSCHEMA: ' + json.dumps(obs.current_schema)
    if obs.schema_diff: schema += '\nDIFF: ' + json.dumps(obs.schema_diff)
    sample = ''
    if obs.data_sample: sample = '\nDATA: ' + json.dumps(obs.data_sample[:3], default=str)
    actions = '\n'.join(f'  {a}' for a in obs.actions_taken[-4:]) or '  (none)'
    return f'STEP {step}/{obs.max_steps} | TASK: {obs.task_id} ({obs.difficulty})\nDESCRIPTION: {obs.description}\nPIPELINE PASSED: {obs.pipeline_passed}\nLAST RESULT: {obs.last_action_result}\nDAG:\n{dag}\nFAILING:\n{failed}\nPASSING: {passed}\nHISTORY:\n{hist}\nACTIONS:\n{actions}{sample}{schema}\nRespond with ONE action JSON.'

def collect_gold(task_ids=['easy','medium', 'hard', 'hard2'], n_ep=10):
    pairs = []
    for tid in task_ids:
        gold = GOLD_ACTIONS.get(tid, [])
        if not gold: continue
        for _ in range(n_ep):
            env = DataPipelineEnv(task_id=tid)
            obs = env.reset()
            for si, ad in enumerate(gold, 1):
                pairs.append((format_obs(obs, si), json.dumps(ad)))
                result = env.step(PipelineAction(**ad))
                obs = result.observation
                if obs.pipeline_passed: break
    return pairs

gold_pairs = collect_gold(n_ep=10)
print(f'Collected {len(gold_pairs)} SFT pairs')

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

sft_texts = [
    tokenizer.apply_chat_template(
        [{'role':'system','content':SYSTEM_PROMPT},
         {'role':'user','content':obs},
         {'role':'assistant','content':act}],
        tokenize=False, add_generation_prompt=False)
    for obs, act in gold_pairs
]
sft_ds = Dataset.from_dict({'text': sft_texts})

SFT_DIR = '/kaggle/working/sft_qwen'

# Extreme VRAM preservation applied to the SFT Training Arguments
print('--- Starting SFT ---')
sft_trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=sft_ds, dataset_text_field='text',
    max_seq_length=MAX_SEQ_LENGTH,
    args=TrainingArguments(
        average_tokens_across_devices=False,
        per_device_train_batch_size=1,       # Drop batch size to 1 minimum
        gradient_accumulation_steps=8,       # Scale gradient acc. steps to compensate
        num_train_epochs=5,
        warmup_ratio=0.1, learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5, optim='adamw_8bit',
        weight_decay=0.01, lr_scheduler_type='cosine',
        output_dir=SFT_DIR, save_steps=50, seed=42,
    ),
)
sft_stats = sft_trainer.train()
print(f'SFT training done. Loss: {sft_stats.training_loss:.4f}')
model.save_pretrained(SFT_DIR)
tokenizer.save_pretrained(SFT_DIR)

# ==============================================================================
# 5. REINFORCEMENT LEARNING ON ENVIRONMENT (GRPO) - STAGE 2
# ==============================================================================
# Instead of static output, model now directly receives reward from the DataPipelineEnv
def parse_action(text):
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.DOTALL).strip()
    if '```' in text:
        text = '\n'.join(l for l in text.split('\n') if not l.strip().startswith('```')).strip()
    start = text.find('{')
    if start == -1: return None
    end = text.rfind('}') + 1
    if end <= start: return None
    try:
        data = json.loads(text[start:end])
        if 'action_type' in data: return PipelineAction(**data)
    except: pass
    return None

def pipeline_reward_fn(completions, **kwargs):
    rewards = []
    for c in completions:
        text = c if isinstance(c, str) else c[0].get('content', '')
        action = parse_action(text)
        if action is None:
            rewards.append(-0.3); continue
        reward = 0.3  # Syntax correctness bonus
        try:
            env = DataPipelineEnv(task_id='hard')
            obs = env.reset()
            result = env.step(action)
            reward += result.reward or 0.0
            if action.action_type == 'compare_schema':
                if result.observation.schema_diff and len(result.observation.schema_diff) > 0:
                    reward += 0.3 # Schema discovery bonus
        except: 
            reward -= 0.2
        rewards.append(float(reward))
    return rewards

from src.tasks import TASKS as _available
grpo_task_ids = [t for t in ['hard', 'hard2'] if t in _available]
grpo_prompts = []

for tid in grpo_task_ids:
    for _ in range(25):
        env = DataPipelineEnv(task_id=tid)
        obs = env.reset()
        chat = tokenizer.apply_chat_template(
            [{'role':'system','content':SYSTEM_PROMPT},
             {'role':'user','content':format_obs(obs, 1)}],
            tokenize=False, add_generation_prompt=True)
        grpo_prompts.append({'prompt': chat})

grpo_ds = Dataset.from_list(grpo_prompts)
print(f'GRPO dataset initialized: {len(grpo_ds)} prompts')

from trl import GRPOConfig, GRPOTrainer

GRPO_DIR = '/kaggle/working/grpo_qwen'
grpo_config = GRPOConfig(
    report_to="none",
    average_tokens_across_devices=False,
    output_dir=GRPO_DIR,
    # VRAM Protections: 4 generations maximum inside the rollouts cache 
    num_generations=4,       
    max_completion_length=200,
    temperature=0.8,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=5e-5,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    # Beta=0.2 acts as a strongly anchored tether to the SFT training rules, resisting severe drift
    beta=0.2, 
    loss_type='grpo',
    logging_steps=5, save_steps=50, seed=42,
    max_prompt_length=MAX_SEQ_LENGTH,
    max_grad_norm=1.0, warmup_ratio=0.05,
)

print(f'--- Starting GRPO training ---')
grpo_trainer = GRPOTrainer(
    model=model, tokenizer=tokenizer,
    reward_funcs=pipeline_reward_fn,
    args=grpo_config, train_dataset=grpo_ds,
)
grpo_stats = grpo_trainer.train()
print(f'GRPO training done. Loss: {grpo_stats.training_loss:.4f}')
model.save_pretrained(GRPO_DIR)
tokenizer.save_pretrained(GRPO_DIR)

# ==============================================================================
# 6. EXPORT / PUSH TO HUGGINGFACE HUB
# ==============================================================================
# Unsloth supports automated LoRA merging into base weights for zero-overhead inference
# 🔥 FIX: remove non-serializable objects from config

LOCAL_MERGED_DIR = '/kaggle/working/qwen-merged-16bit'
print(f'Saving completely merged 16-bit model locally to {LOCAL_MERGED_DIR}...')

# 🔥 FIX (ADD THIS)
model.config.__dict__ = {
    k: v for k, v in model.config.__dict__.items()
    if not callable(v)
}

model.save_pretrained_merged(
    LOCAL_MERGED_DIR,
    tokenizer,
    save_method='merged_16bit'
)
HF_REPO = 'Abhinav-hf/data-pipeline-incident-qwen-grpo'
if HF_TOKEN:
    print(f'Pushing seamlessly compiled model to Hub: {HF_REPO}')
    model.push_to_hub_merged(HF_REPO, tokenizer, save_method='merged_16bit', token=HF_TOKEN)
    print(f'Done! Model available at: https://huggingface.co/{HF_REPO}')
else:
    print('No HF_TOKEN detected — skipping Hub upload. Local save complete.')
