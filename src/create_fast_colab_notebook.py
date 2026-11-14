import json

def generate_fast_colab_notebook():
    cells = []
    
    # Cell 1: Markdown title & explanation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# ⚡ The Liar's Chatroom - High-Speed Colab Solver (Optimized for T4)\n",
            "\n",
            "### Why was it taking 37s before?\n",
            "1. **T4 has NO bfloat16 hardware**: Turing GPUs (T4) must use `float16` (FP16), otherwise PyTorch falls back to slow software emulation.\n",
            "2. **14B dequantization overhead**: Generating 150+ tokens per sample sequentially on a 14B model takes ~35s.\n",
            "3. **Optimization applied here**:\n",
            "   - Switched compute dtype to native **`torch.float16`**.\n",
            "   - Switched to **`Qwen/Qwen2.5-7B-Instruct`** (or `DeepSeek-R1-Distill-Qwen-7B`), which runs at **~40 tokens/sec** on T4.\n",
            "   - Constrained generation to **`max_new_tokens=25`** (only 1 concise sentence + ANSWER: [A-D]).\n",
            "   - **Expected speed: ~0.6s - 0.9s per case** (~35 mins for all 3,500 cases on free T4!)."
        ]
    })
    
    # Cell 2: Dependencies
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "!pip install -q -U transformers accelerate bitsandbytes pandas tqdm"
        ]
    })
    
    # Cell 3: Data Upload Check
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "from google.colab import files\n",
            "import pandas as pd\n",
            "\n",
            "if not os.path.exists('test.csv'):\n",
            "    print('Please upload test.csv:')\n",
            "    uploaded = files.upload()\n",
            "else:\n",
            "    print('Found existing test.csv!')\n",
            "    \n",
            "test_df = pd.read_csv('test.csv')\n",
            "print(f'Test dataset loaded: {len(test_df)} cases.')"
        ]
    })
    
    # Cell 4: High-speed solver logic
    fast_code = '''import os
import re
import gc
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------------------------------------------------
# 1. Parsing & Filler Filtering
# -------------------------------------------------------------
def parse_speakers(speakers_str):
    l2n = {}
    for part in speakers_str.split(';'):
        if '=' in part:
            k, v = part.split('=', 1)
            l2n[k.strip()] = v.strip()
    return l2n

COMMON_FILLERS = {
    "this is getting muddled.", "this is all rather confusing.", "this is a muddle.",
    "this is hard to follow.", "honestly this is hard to follow.", "i'd like a clear answer here.",
    "i'd like a clear answer on this.", "let's keep this thread brief.", "let's keep it short.",
    "let's wrap this up.", "thanks, everyone, for checking.", "thanks, everyone.",
    "thanks to everyone for checking.", "we could really use a faster van.", "can we get a plain answer."
}

def is_filler_message(text):
    clean = text.strip()
    if '?' in clean:
        return True
    clean_lower = clean.lower()
    if clean_lower in COMMON_FILLERS:
        return True
    if any(clean_lower.startswith(p) for p in ["this is ", "i'd like ", "let's ", "thanks"]):
        if len(clean.split()) < 8 and not re.search(r'\\b(\\d+|am|pm|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\\b', clean_lower):
            return True
    return False

def parse_chat(chat_str, letter_to_name):
    messages = []
    lines = [l.strip() for l in chat_str.split('\\n') if l.strip()]
    for line in lines:
        m = re.match(r'^(\\d+)\\.\\s*([A-D]):\\s*(.*)$', line)
        if not m:
            continue
        messages.append({
            'num': int(m.group(1)),
            'sender': m.group(2),
            'text': m.group(3).strip(),
            'is_filler': is_filler_message(m.group(3).strip())
        })
    return messages

def format_prompt(row):
    l2n = parse_speakers(row['speakers'])
    msgs = parse_chat(row['chat'], l2n)
    
    claims = {'A': [], 'B': [], 'C': [], 'D': []}
    for m in msgs:
        if not m['is_filler']:
            claims[m['sender']].append(f"- {m['text']}")
            
    claims_text = []
    for s in ['A', 'B', 'C', 'D']:
        name = l2n.get(s, s)
        if claims[s]:
            claims_text.append(f"Speaker {s} ({name}) claims: " + " ".join(claims[s]))
        else:
            claims_text.append(f"Speaker {s} ({name}) made no claims.")
            
    claims_block = "\\n".join(claims_text)
    
    prompt = f"""Scenario: {row['scenario']}
Speakers: {row['speakers']}
Evidence: {row['evidence']}

Claims made by speakers:
{claims_block}

Rule: Exactly one speaker made a claim that contradicts the verified evidence. Speakers who made no claims are innocent.
Identify who lied. Give a 1-sentence reason, then output ANSWER: [A/B/C/D]."""
    return prompt

def extract_answer(resp, valid_candidates):
    m = re.search(r'ANSWER:\\s*([A-D])\\b', resp, re.IGNORECASE)
    if m and (not valid_candidates or m.group(1).upper() in valid_candidates):
        return m.group(1).upper()
        
    m = re.search(r'(?:speaker|liar is|answer is)\\s*([A-D])\\b', resp, re.IGNORECASE)
    if m and (not valid_candidates or m.group(1).upper() in valid_candidates):
        return m.group(1).upper()
        
    matches = re.findall(r'\\b([A-D])\\b', resp)
    if matches:
        for ans in reversed(matches):
            if not valid_candidates or ans in valid_candidates:
                return ans
                
    return sorted(list(valid_candidates))[0] if valid_candidates else 'A'

# -------------------------------------------------------------
# 2. Fast Model Loading (7B 4-bit with native FP16 for T4)
# -------------------------------------------------------------
assert torch.cuda.is_available(), "Please enable GPU in Runtime -> Change runtime type -> T4 GPU!"
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Qwen2.5-7B is state-of-the-art and 5x-8x faster than 14B on T4
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # CRITICAL: T4 requires float16, NOT bfloat16
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

print(f"Loading {MODEL_ID} in fast 4-bit FP16 mode...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# -------------------------------------------------------------
# 3. High-Speed Inference Loop
# -------------------------------------------------------------
test_df = pd.read_csv('test.csv')
print(f"Running high-speed inference on {len(test_df)} cases...")

predictions = []
device = "cuda"

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Solving Cases"):
    cid = row['id']
    l2n = parse_speakers(row['speakers'])
    msgs = parse_chat(row['chat'], l2n)
    valid_candidates = set(m['sender'] for m in msgs if not m['is_filler'])
    
    prompt_content = format_prompt(row)
    messages = [
        {"role": "system", "content": "You are a precise logic solver. You identify the one speaker whose claim is impossible according to the evidence."},
        {"role": "user", "content": prompt_content}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,       # Fast concise output: ~20-25 tokens max!
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
    new_toks = outputs[0][inputs['input_ids'].shape[1]:]
    resp = tokenizer.decode(new_toks, skip_special_tokens=True)
    pred = extract_answer(resp, valid_candidates)
    
    predictions.append({'id': cid, 'label': pred})

df_sub = pd.DataFrame(predictions)
df_sub.to_csv('submission.csv', index=False)
print("\\nDone! Saved submission.csv.")
'''

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in fast_code.splitlines()]
    })
    
    # Cell 5: Auto-download & verify
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "from google.colab import files\n",
            "\n",
            "df_sub = pd.read_csv('submission.csv')\n",
            "print('Total rows:', len(df_sub))\n",
            "print('Null values:', df_sub.isnull().sum().to_dict())\n",
            "print('Label distribution:\\n', df_sub['label'].value_counts())\n",
            "\n",
            "files.download('submission.csv')"
        ]
    })
    
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    out_file = 'colab_liars_chatroom_solution.ipynb'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Updated fast notebook: {out_file}")

if __name__ == '__main__':
    generate_fast_colab_notebook()
