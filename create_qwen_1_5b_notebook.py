import json
import re
import os

with open('run_on_kaggle/run_on_kaggle_qwen_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'source' not in cell: continue
    
    # If the source is a string instead of a list, make it a list for easier processing
    source = cell['source']
    if isinstance(source, str):
        source = [source]
        
    new_source = []
    for line in source:
        # Title and markdown updates
        line = line.replace('Qwen2.5-VL-3B-Instruct', 'Qwen2.5-1.5B-Instruct')
        line = line.replace('Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct')
        
        # Imports and class updates
        if 'Qwen2_5_VLForConditionalGeneration' in line or 'Qwen2VLForConditionalGeneration' in line or '_VL_CLASS' in line:
            if 'from transformers import' in line:
                line = "    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n"
            elif 'model = _VL_CLASS' in line:
                line = "model = AutoModelForCausalLM.from_pretrained(\n"
            else:
                continue # Skip the try-except logic for VL class fallback
        
        # Processor to Tokenizer
        if 'processor = AutoProcessor.from_pretrained' in line:
            line = "tokenizer = AutoTokenizer.from_pretrained(\n"
        elif 'min_pixels' in line or 'max_pixels' in line:
            continue # Remove VL specific arguments
        
        # MAX_STEPS and MAX_TOKENS updates
        if "MAX_TOKENS  = int(os.getenv('MAX_TOKENS',   '1024'))" in line:
            line = "MAX_TOKENS  = int(os.getenv('MAX_TOKENS',   '1024'))\n"
        if "MAX_STEPS   = int(os.getenv('MAX_STEPS', '30'))" in line:
            line = "MAX_STEPS   = int(os.getenv('MAX_STEPS', '100'))\n"
            
        # _call_model updates
        line = line.replace('processor.apply_chat_template', 'tokenizer.apply_chat_template')
        line = line.replace('processor(', 'tokenizer(')
        line = line.replace('processor.tokenizer.eos_token_id', 'tokenizer.eos_token_id')
        line = line.replace('processor.tokenizer.decode', 'tokenizer.decode')
        
        new_source.append(line)
        
    # Re-assemble the source string if it was originally a string
    if isinstance(cell['source'], str):
        cell['source'] = "".join(new_source)
    else:
        cell['source'] = new_source

with open('run_on_kaggle/run_on_kaggle_qwen_1.5b.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Created run_on_kaggle/run_on_kaggle_qwen_1.5b.ipynb")
