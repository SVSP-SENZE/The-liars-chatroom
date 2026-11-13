from chat_parser import parse_speakers, parse_chat

def format_prompt_for_llm(row, include_thinking_instructions=True):
    """
    Constructs a high-precision prompt for an open-weight LLM (e.g. Qwen 2.5 / DeepSeek-R1 / Llama 3.1).
    """
    scenario = row['scenario']
    speakers = row['speakers']
    evidence = row['evidence']
    chat = row['chat']
    
    l2n, _ = parse_speakers(speakers)
    parsed_msgs = parse_chat(chat, l2n)
    
    # Group claims by speaker
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
    
    prompt = f"""You are an expert logical verifier. Your task is to identify the single speaker who lied.

BACKGROUND & SCENARIO:
{scenario}

SPEAKERS:
{speakers}

VERIFIED EVIDENCE (GROUND TRUTH FACTS):
{evidence}

CANDIDATE CLAIMS MADE IN THE CHAT:
{formatted_claims}

REASONING RULES & GUIDELINES:
1. Only the Evidence and Speaker Table are true.
2. Exactly ONE speaker sent a message that contradicts the evidence.
3. A claim contradicts the evidence ONLY if it is strictly impossible / cannot be true together with the evidence facts.
4. A claim that is merely unverified, unconfirmed, or suspicious is NOT a contradiction.
5. Watch boundaries: ranges like "between 8:00 AM and 9:00 AM" include both endpoints (8:00 and 9:00).
6. Calendar days and clock arithmetic must be calculated with precision.
7. Any speaker who made NO factual claims cannot be the liar.

TASK:
Check each speaker's claims against the evidence step-by-step.
Identify the single claim that is logically or physically impossible according to the evidence.
Conclude with the final answer as the single uppercase letter of the speaker (A, B, C, or D).

Format your response as:
REASONING: <brief step-by-step verification>
ANSWER: <A, B, C, or D>"""

    return prompt
