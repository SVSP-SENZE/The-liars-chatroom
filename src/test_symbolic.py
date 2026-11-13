import re
import pandas as pd
from datetime import datetime
from chat_parser import parse_speakers, parse_chat

def parse_time_str(t_str):
    """Parses '11:30 AM', '1:10 PM', '12:00 PM' into minutes from midnight."""
    m = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', t_str, re.IGNORECASE)
    if not m:
        return None
    hh, mm, mer = int(m.group(1)), int(m.group(2)), m.group(3).upper()
    if mer == 'PM' and hh != 12:
        hh += 12
    elif mer == 'AM' and hh == 12:
        hh = 0
    return hh * 60 + mm

def check_time_window_contradiction(evidence, claim):
    """
    Checks common patterns like:
    - 'From 11:15 AM to 11:45 AM the water supply was shut off' / 'Nobody could enter ... between 9:00 AM and 11:00 AM'
    and claims like:
    - 'I made a coffee ... at 11:30 AM'
    - 'Between 9:45 AM and 12:45 PM I didn't leave the office'
    """
    ev_lower = evidence.lower()
    cl_lower = claim.lower()
    
    # Pattern: Shut off / closed / nobody could ... from X to Y / between X and Y
    m_range = re.search(r'(?:from|between)\s*(\d{1,2}:\d{2}\s*(?:am|pm))\s*(?:to|and)\s*(\d{1,2}:\d{2}\s*(?:am|pm))', ev_lower)
    if m_range:
        t_start = parse_time_str(m_range.group(1))
        t_end = parse_time_str(m_range.group(2))
        
        # Check claim time: "at 11:30 AM"
        m_claim_time = re.search(r'at\s*(\d{1,2}:\d{2}\s*(?:am|pm))', cl_lower)
        if m_claim_time and t_start is not None and t_end is not None:
            t_claim = parse_time_str(m_claim_time.group(1))
            if t_claim is not None and t_start <= t_claim <= t_end:
                # If evidence says shut off / could not / closed during this time:
                if any(k in ev_lower for k in ['shut off', 'closed', 'nobody could', 'locked', 'could only make', 'only while']):
                    if any(k in cl_lower for k in ['made a coffee', 'entered', 'inside', 'boarded', 'swam', 'unlocked']):
                        return True
                        
    return False

# Test on train set
df = pd.read_csv('data/train_folds.csv')
detected = 0
correct = 0

for idx, row in df.iterrows():
    l2n, _ = parse_speakers(row['speakers'])
    msgs = parse_chat(row['chat'], l2n)
    
    for m in msgs:
        if m['is_filler']:
            continue
        if check_time_window_contradiction(row['evidence'], m['text']):
            detected += 1
            if m['sender'] == row['label']:
                correct += 1
            break

print(f"Simple heuristic flagged: {detected} cases. Correct: {correct} ({correct/max(1, detected)*100:.1f}%)")
