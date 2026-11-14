"""
prompt_v2.py
High-fidelity structured prompt with deterministic temporal injection.
Designed for DeepSeek-R1-Distill, Qwen2.5-14B/32B, and similar reasoning LLMs.
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from chat_parser import parse_speakers, parse_chat
from temporal_preprocessor import compute_temporal_facts


def build_prompt(row, include_chat_verbatim=True) -> dict:
    """
    Returns a dict with:
      - 'system': system prompt string
      - 'user': user prompt string
      - 'valid_candidates': set of speaker letters that made factual claims
      - 'computed_facts': the injected temporal string
    """
    scenario   = row['scenario']
    speakers   = row['speakers']
    evidence   = row['evidence']
    chat_raw   = row['chat']

    l2n, n2l = parse_speakers(speakers)
    msgs = parse_chat(chat_raw, l2n)

    # Deterministic temporal injection
    computed_facts = compute_temporal_facts(evidence, chat_raw)

    # Group factual claims by sender (normalized: replace I/my with speaker name)
    speaker_claims = {s: [] for s in ['A', 'B', 'C', 'D']}
    for m in msgs:
        if not m['is_filler']:
            # Normalize first-person pronouns to speaker name for clarity
            txt = m['normalized_text']
            speaker_claims[m['sender']].append(
                f"  [{m['num']}] {txt}"
            )

    valid_candidates = set(s for s in ['A', 'B', 'C', 'D'] if speaker_claims[s])

    # Build the claims block
    claims_block_lines = []
    for s in ['A', 'B', 'C', 'D']:
        name = l2n.get(s, s)
        if speaker_claims[s]:
            claims_block_lines.append(f"Speaker {s} = {name}:")
            claims_block_lines.extend(speaker_claims[s])
        else:
            claims_block_lines.append(
                f"Speaker {s} = {name}: [NO FACTUAL CLAIMS — only asked questions or made comments. CANNOT be the liar.]"
            )
        claims_block_lines.append("")

    claims_block = "\n".join(claims_block_lines).strip()

    # Optional verbatim chat (helps LLM anchor on exact wording)
    chat_section = ""
    if include_chat_verbatim:
        chat_section = f"\nFULL CHAT (verbatim, for reference):\n{chat_raw}\n"

    system_prompt = (
        "You are a meticulous logical verifier. "
        "Your task is to find the single speaker whose message makes a claim that is "
        "IMPOSSIBLE given the verified evidence. You must reason step-by-step.\n\n"
        "Key rules:\n"
        "- Only the EVIDENCE is ground truth. Chat messages are claims to be verified.\n"
        "- A contradiction means a claim is IMPOSSIBLE, not merely unverified or unlikely.\n"
        "- 'Between X and Y' and 'from X to Y' INCLUDE both endpoints X and Y.\n"
        "- Watch carefully: if Speaker D says 'I saw Goran at the stadium', "
        "and Goran could not be there, then Speaker D lied — the SENDER is the liar.\n"
        "- Speakers who made NO factual claims cannot be the liar.\n"
        "- Do precise arithmetic for times, dates, durations, and counts."
    )

    user_prompt = f"""SCENARIO: {scenario}

SPEAKERS: {speakers}

VERIFIED EVIDENCE (treat every word as absolute truth):
{evidence}
{computed_facts}
FACTUAL CLAIMS (questions and filler comments removed):
{claims_block}
{chat_section}
TASK:
Check each speaker's factual claims against the evidence one-by-one.
Identify which single claim is IMPOSSIBLE according to the evidence.
State your reasoning concisely, then give your final answer.

End your response with EXACTLY this line:
FINAL ANSWER: [single uppercase letter A, B, C, or D]"""

    return {
        "system": system_prompt,
        "user": user_prompt,
        "valid_candidates": valid_candidates,
        "computed_facts": computed_facts,
    }


def extract_answer(response: str, valid_candidates: set) -> str:
    """
    Robust extraction of A/B/C/D from model response.
    Priority: explicit FINAL ANSWER tag > inline patterns > last letter found.
    """
    # 1. Explicit tag (highest priority)
    for pattern in [
        r'FINAL ANSWER:\s*([A-D])\b',
        r'ANSWER:\s*([A-D])\b',
        r'Answer:\s*([A-D])\b',
    ]:
        m = re.search(pattern, response)
        if m and (not valid_candidates or m.group(1).upper() in valid_candidates):
            return m.group(1).upper()

    # 2. Explicit statement patterns
    for pattern in [
        r'(?:the liar is|lied is|speaker who lied is|liar:|lied:)\s*(?:speaker\s*)?([A-D])\b',
        r'(?:speaker|answer is)\s+([A-D])\b',
        r'\bthe (?:answer|liar|contradiction) is\s+(?:speaker\s+)?([A-D])\b',
        r'\b([A-D])\s+(?:lied|is the liar|sent the lie|made the false)',
    ]:
        m = re.search(pattern, response, re.IGNORECASE)
        if m and (not valid_candidates or m.group(1).upper() in valid_candidates):
            return m.group(1).upper()

    # 3. Last standalone letter in the response (last resort)
    all_letters = re.findall(r'\b([A-D])\b', response)
    if all_letters:
        for letter in reversed(all_letters):
            if not valid_candidates or letter in valid_candidates:
                return letter

    # 4. Fallback to first valid candidate
    if valid_candidates:
        return sorted(valid_candidates)[0]
    return 'A'


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('dataset/extracted/train.csv')
    for i in [2, 10, 25]:
        row = df.iloc[i]
        result = build_prompt(row)
        print("=" * 70)
        print("SYSTEM:", result['system'][:200])
        print("USER:", result['user'])
        print("VALID:", result['valid_candidates'])
        print()
