"""
Kaggle / Cloud GPU Turnkey Solver for The Liar's Chatroom
Model: Open-Weight Reasoning Model (e.g. Qwen/Qwen2.5-14B-Instruct or DeepSeek-R1-Distill-Qwen-14B)
Fully compliant with all competition rules:
- 100% open-weight model (licensed up to 70B parameters)
- No closed/hosted APIs (no OpenAI, no Anthropic, no Google Gemini API)
- Pure offline/local inference
- Outputs valid submission.csv
"""

import os
import re
import gc
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------------------------------------------------
# 1. Chat Parsing & Candidate Extraction
# -------------------------------------------------------------
def parse_speakers(speakers_str):
    letter_to_name = {}
    for part in speakers_str.split(';'):
        if '=' in part:
            letter, name = part.split('=', 1)
            letter_to_name[letter.strip()] = name.strip()
    return letter_to_name

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
        if len(clean.split()) < 8 and not re.search(r'\b(\d+|am|pm|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', clean_lower):
            return True
    return False

def parse_chat(chat_str, letter_to_name):
    messages = []
    lines = [l.strip() for l in chat_str.split('\n') if l.strip()]
    for line in lines:
        m = re.match(r'^(\d+)\.\s*([A-D]):\s*(.*)$', line)
        if not m:
            continue
        num = int(m.group(1))
        sender = m.group(2)
        text = m.group(3).strip()
        filler = is_filler_message(text)
        messages.append({
            'num': num,
            'sender': sender,
            'text': text,
            'is_filler': filler
        })
    return messages

def format_prompt(row):
    scenario = row['scenario']
    speakers = row['speakers']
    evidence = row['evidence']
    chat = row['chat']
    
    l2n = parse_speakers(speakers)
    parsed_msgs = parse_chat(chat, l2n)
    
    speaker_claims = {'A': [], 'B': [], 'C': [], 'D': []}
    for m in parsed_msgs:
        if not m['is_filler']:
            speaker_claims[m['sender']].append(f"- Message {m['num']}: \"{m['text']}\"")
            
    claims_text = []
    for s in ['A', 'B', 'C', 'D']:
        name = l2n.get(s, s)
        if len(speaker_claims[s]) > 0:
            c_str = "\n".join(speaker_claims[s])
            claims_text.append(f"Speaker {s} ({name}) claims:\n{c_str}")
        else:
            claims_text.append(f"Speaker {s} ({name}) made NO factual claims (only questions/comments).")
            
    formatted_claims = "\n\n".join(claims_text)
    
    prompt = f"""You are a logical verification engine. Exactly one speaker in this group chat lied by making a statement that directly contradicts the verified evidence.

BACKGROUND:
{scenario}

SPEAKERS:
{speakers}

VERIFIED EVIDENCE (ABSOLUTE GROUND TRUTH):
{evidence}

CANDIDATE CLAIMS MADE IN CHAT:
{formatted_claims}

CRITICAL RULES:
1. Only the Evidence and Speaker Table are true.
2. Exactly ONE speaker sent a message that contradicts the evidence.
3. A claim contradicts the evidence ONLY if it is impossible given the evidence facts.
4. Unverified or suspicious claims are NOT contradictions.
5. "Between X and Y" includes both endpoints X and Y.
6. A speaker who made NO factual claims cannot be the liar.

TASK:
Verify each claim against the evidence step-by-step.
Identify the single speaker whose claim is logically or physically impossible.
Conclude with the final answer as the single uppercase letter of the speaker (A, B, C, or D).

Format your response as:
REASONING: <concise logical verification>
ANSWER: <A, B, C, or D>"""

    return prompt

def extract_answer(response_text, valid_candidates):
    # Search for explicit ANSWER: [A-D]
    m = re.search(r'ANSWER:\s*([A-D])\b', response_text, re.IGNORECASE)
    if m:
        ans = m.group(1).upper()
        if ans in valid_candidates:
            return ans
            
    m = re.search(r'(?:speaker|liar is|answer is)\s*([A-D])\b', response_text, re.IGNORECASE)
    if m:
        ans = m.group(1).upper()
        if ans in valid_candidates:
            return ans
            
    matches = re.findall(r'\b([A-D])\b', response_text)
    if matches:
        for ans in reversed(matches):
            if ans in valid_candidates:
                return ans
                
    return sorted(list(valid_candidates))[0] if valid_candidates else 'A'

# -------------------------------------------------------------
# 2. Main Execution
# -------------------------------------------------------------
def main():
    MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"  # or "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" or "Qwen/Qwen2.5-7B-Instruct"
    
    # Path configuration (supports both local and Kaggle environments)
    if os.path.exists('/kaggle/input'):
        DATA_DIR = '/kaggle/input/the-liars-chatroom'
        OUT_DIR = '/kaggle/working'
    else:
        DATA_DIR = 'dataset/extracted'
        OUT_DIR = 'submissions'
        
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    
    print(f"Loading data from {DATA_DIR}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    # Quantization config (4-bit for 14B models on 16GB VRAM)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    ) if torch.cuda.is_available() else None
    
    print(f"Loading model: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    model.eval()
    
    # ---------------------------------------------------------
    # Quick Validation Check (First 100 samples of train)
    # ---------------------------------------------------------
    print("\n--- Running Quick Validation on 100 Train Cases ---")
    val_subset = train_df.head(100)
    correct = 0
    
    for idx, row in tqdm(val_subset.iterrows(), total=len(val_subset), desc="Validating"):
        l2n = parse_speakers(row['speakers'])
        msgs = parse_chat(row['chat'], l2n)
        valid_candidates = set(m['sender'] for m in msgs if not m['is_filler'])
        
        prompt_content = format_prompt(row)
        messages = [
            {"role": "system", "content": "You are a precise logical reasoning assistant."},
            {"role": "user", "content": prompt_content}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        new_toks = outputs[0][inputs['input_ids'].shape[1]:]
        resp = tokenizer.decode(new_toks, skip_special_tokens=True)
        pred = extract_answer(resp, valid_candidates)
        
        if pred == row['label']:
            correct += 1
            
    val_acc = correct / len(val_subset) * 100
    print(f"\n>>> Validation Accuracy on 100 cases: {correct}/{len(val_subset)} ({val_acc:.2f}%) <<<\n")
    
    # ---------------------------------------------------------
    # Full Test Set Inference (3,500 cases)
    # ---------------------------------------------------------
    print(f"--- Running Inference on {len(test_df)} Test Cases ---")
    predictions = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Testing"):
        cid = row['id']
        l2n = parse_speakers(row['speakers'])
        msgs = parse_chat(row['chat'], l2n)
        valid_candidates = set(m['sender'] for m in msgs if not m['is_filler'])
        
        prompt_content = format_prompt(row)
        messages = [
            {"role": "system", "content": "You are a precise logical reasoning assistant."},
            {"role": "user", "content": prompt_content}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        new_toks = outputs[0][inputs['input_ids'].shape[1]:]
        resp = tokenizer.decode(new_toks, skip_special_tokens=True)
        pred = extract_answer(resp, valid_candidates)
        
        predictions.append({'id': cid, 'label': pred})
        
    df_sub = pd.DataFrame(predictions)
    out_csv = os.path.join(OUT_DIR, 'submission.csv')
    df_sub.to_csv(out_csv, index=False)
    print(f"\nFinal submission saved to: {out_csv}")
    print("\nSubmission summary:")
    print(df_sub['label'].value_counts(normalize=True))

if __name__ == '__main__':
    main()
