import re

def parse_speakers(speakers_str):
    """
    Parses 'A = Nina; B = Ade; C = Taro; D = Lena'
    Returns:
        letter_to_name: {'A': 'Nina', 'B': 'Ade', 'C': 'Taro', 'D': 'Lena'}
        name_to_letter: {'Nina': 'A', 'Ade': 'B', 'Taro': 'C', 'Lena': 'D'}
    """
    letter_to_name = {}
    name_to_letter = {}
    for part in speakers_str.split(';'):
        if '=' in part:
            letter, name = part.split('=', 1)
            letter = letter.strip()
            name = name.strip()
            letter_to_name[letter] = name
            name_to_letter[name.lower()] = letter
    return letter_to_name, name_to_letter

COMMON_OPINION_STARTS = [
    "this is ", "i'd like ", "let's ", "thanks", "honestly this is ", "we could really ",
    "i think ", "in my opinion", "that is ", "it's getting ", "it is getting "
]

def is_filler_message(text):
    """
    Returns True if the message is clearly a question or pure conversational filler/opinion.
    Rules:
    - Questions (contains '?') are ALWAYS fillers.
    - Meta-statements without any concrete claims are fillers.
    """
    clean = text.strip()
    if '?' in clean:
        return True
    
    clean_lower = clean.lower()
    
    # Check for well-known generic conversational comments
    generic_fillers = {
        "this is getting muddled.",
        "this is all rather confusing.",
        "this is a muddle.",
        "this is hard to follow.",
        "honestly this is hard to follow.",
        "i'd like a clear answer here.",
        "i'd like a clear answer on this.",
        "let's keep this thread brief.",
        "let's keep it short.",
        "let's wrap this up.",
        "thanks, everyone, for checking.",
        "thanks, everyone.",
        "thanks to everyone for checking.",
        "we could really use a faster van.",
        "can we get a plain answer.",
    }
    if clean_lower in generic_fillers:
        return True
        
    for op in COMMON_OPINION_STARTS:
        if clean_lower.startswith(op) and len(clean.split()) < 8 and not re.search(r'\b(\d+|am|pm|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', clean_lower):
            return True
            
    return False

def parse_chat(chat_str, letter_to_name):
    """
    Parses chat string into a list of message dicts:
    [{
        'num': int,
        'sender': 'A'|'B'|'C'|'D',
        'sender_name': str,
        'text': str,
        'is_filler': bool,
        'normalized_text': str  # replaces 'I' / 'my' with speaker's name for NLI
    }]
    """
    messages = []
    lines = [l.strip() for l in chat_str.split('\n') if l.strip()]
    
    for line in lines:
        m = re.match(r'^(\d+)\.\s*([A-D]):\s*(.*)$', line)
        if not m:
            continue
        num = int(m.group(1))
        sender = m.group(2)
        text = m.group(3).strip()
        sender_name = letter_to_name.get(sender, sender)
        
        filler = is_filler_message(text)
        
        # Create a normalized version of text for NLI
        # E.g. "I entered at 10 AM" -> "Nina (A) entered at 10 AM"
        # Be careful with word boundaries
        norm = re.sub(r'\bI\b', f"{sender_name} ({sender})", text)
        norm = re.sub(r'\bmy\b', f"{sender_name}'s", norm, flags=re.IGNORECASE)
        norm = re.sub(r'\bme\b', f"{sender_name} ({sender})", norm, flags=re.IGNORECASE)
        norm = re.sub(r'\bmyself\b', f"{sender_name} ({sender})", norm, flags=re.IGNORECASE)
        
        messages.append({
            'num': num,
            'sender': sender,
            'sender_name': sender_name,
            'text': text,
            'is_filler': filler,
            'normalized_text': norm
        })
        
    return messages
