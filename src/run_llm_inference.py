import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from prompt_builder import format_prompt_for_llm
from chat_parser import parse_speakers, parse_chat

def extract_answer(response_text, valid_candidates=None):
    """
    Extracts the speaker label (A, B, C, or D) from model output with fallback logic.
    """
    # 1. Look for explicit pattern: ANSWER: X
    m = re.search(r'ANSWER:\s*([A-D])\b', response_text, re.IGNORECASE)
    if m:
        ans = m.group(1).upper()
        if valid_candidates is None or ans in valid_candidates:
            return ans
            
    # 2. Look for "the answer is X" or "speaker X lied" or "liar is X"
    m = re.search(r'(?:answer is|speaker|liar is|contradiction is)\s*([A-D])\b', response_text, re.IGNORECASE)
    if m:
        ans = m.group(1).upper()
        if valid_candidates is None or ans in valid_candidates:
            return ans
            
    # 3. Look for the last standalone [A-D] in the text
    matches = re.findall(r'\b([A-D])\b', response_text)
    if matches:
        for ans in reversed(matches):
            if valid_candidates is None or ans in valid_candidates:
                return ans
                
    # 4. Fallback: return first candidate
    if valid_candidates:
        return sorted(list(valid_candidates))[0]
    return 'A'

def run_inference(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    test_df = pd.read_csv(args.test_csv)
    if args.max_samples > 0:
        test_df = test_df.iloc[:args.max_samples]
    print(f"Loaded {len(test_df)} test cases.")
    
    print(f"Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Inference device:", device)
    
    # Load model with optional 4-bit / 8-bit quantization
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    elif args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    model.eval()
    
    results = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        case_id = row['id']
        l2n, _ = parse_speakers(row['speakers'])
        msgs = parse_chat(row['chat'], l2n)
        valid_candidates = set(m['sender'] for m in msgs if not m['is_filler'])
        
        user_prompt = format_prompt_for_llm(row)
        
        # Apply chat template if supported
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": "You are an expert logical verification engine. Carefully identify the one speaker who contradicted the evidence."},
                {"role": "user", "content": user_prompt}
            ]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = user_prompt
            
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id
            )
            
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        pred_label = extract_answer(response, valid_candidates)
        results.append({
            'id': case_id,
            'label': pred_label,
            'raw_response': response
        })
        
    df_res = pd.DataFrame(results)
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_res[['id', 'label']].to_csv(args.output_csv, index=False)
    print(f"Saved submission to {args.output_csv}")
    
    # Save reasoning logs for analysis
    logs_path = args.output_csv.replace('.csv', '_reasoning_logs.csv')
    df_res.to_csv(logs_path, index=False)
    print(f"Saved reasoning logs to {logs_path}")
    
    print("\nLabel Distribution:")
    print(df_res['label'].value_counts(normalize=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_csv', type=str, default='dataset/extracted/test.csv')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--output_csv', type=str, default='submissions/sub_llm.csv')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    args = parser.parse_args()
    run_inference(args)
