import json
import os

path = r"c:\Users\micro\Desktop\Abhinav college\Resources\Sem 8\Meta_Hackathon\Meta_hackathon\train_grpo\training-grpo-qwen.ipynb"

try:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except Exception as e:
    print('Failed to load JSON:', e)
    exit(1)

modified = False
hard_code = """    'hard': [
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
"""

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        text = "".join(cell['source'])
        old_text = text
        
        # update collection to include hard / hard2
        if "def collect_gold(task_ids=['easy','medium']" in text:
            text = text.replace("def collect_gold(task_ids=['easy','medium']", "def collect_gold(task_ids=['easy','medium', 'hard', 'hard2']")
            
        if "GOLD_ACTIONS = {" in text and "hard2" not in text:
            # We insert it right before the closing bracket of GOLD_ACTIONS
            text = text.replace("    ],\n}\n", "    ],\n" + hard_code + "}\n")

        if "grpo_config = GRPOConfig(" in text:
            text = text.replace("beta=0.1,", "beta=0.2,")

        if text != old_text:
            modified = True
            cell['source'] = [l + '\\n' for l in text.split('\\n')][:-1]

if modified:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print('Notebook saved.')
else:
    print('No modifications were needed.')
