"""
temporal_preprocessor.py
Deterministic date/time reasoning injected into prompts.
LLMs hallucinate day-of-week and interval math — this module makes it exact.
"""

import re
from datetime import date, datetime, timedelta

MONTH_MAP = {
    'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
    'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
    'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
    'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
}

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def parse_time_to_minutes(t_str: str):
    """Convert '11:30 AM' or '2:05 PM' to minutes since midnight."""
    m = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', t_str, re.IGNORECASE)
    if not m:
        return None
    h, mi, mer = int(m.group(1)), int(m.group(2)), m.group(3).upper()
    if mer == 'PM' and h != 12:
        h += 12
    elif mer == 'AM' and h == 12:
        h = 0
    return h * 60 + mi

def minutes_to_hhmm(mins: int) -> str:
    h = mins // 60
    mi = mins % 60
    mer = 'AM' if h < 12 else 'PM'
    if h == 0: h = 12
    elif h > 12: h -= 12
    return f"{h}:{mi:02d} {mer}"

def extract_all_dates_in_text(text: str):
    """
    Extract all 'D Month YYYY' or 'Month D YYYY' or 'D April 2026' style dates.
    Returns list of (original_string, date_object)
    """
    results = []
    # Pattern: "14 April 2026" or "April 14 2026" or "14 Apr 2026"
    patterns = [
        r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})',
        r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})\s+(\d{4})',
    ]
    text_lower = text.lower()
    
    for pat_idx, pattern in enumerate(patterns):
        for m in re.finditer(pattern, text_lower):
            try:
                if pat_idx == 0:
                    day = int(m.group(1))
                    month = MONTH_MAP.get(m.group(2).lower(), 0)
                    year = int(m.group(3))
                else:
                    month = MONTH_MAP.get(m.group(1).lower(), 0)
                    day = int(m.group(2))
                    year = int(m.group(3))
                
                if month == 0:
                    continue
                    
                d = date(year, month, day)
                orig = m.group(0)
                results.append((orig, d))
            except (ValueError, KeyError):
                continue
                
    return results

def extract_all_times_in_text(text: str):
    """Extract all HH:MM AM/PM style times."""
    results = []
    for m in re.finditer(r'\d{1,2}:\d{2}\s*(?:AM|PM)', text, re.IGNORECASE):
        t = parse_time_to_minutes(m.group(0))
        if t is not None:
            results.append((m.group(0), t))
    return results

def compute_temporal_facts(evidence: str, chat: str = '') -> str:
    """
    Given evidence (and optionally chat), compute deterministic temporal facts.
    Returns a string of verified facts to inject into the prompt.
    """
    facts = []
    combined = evidence + '\n' + chat
    
    # ----- 1. Compute day-of-week for all dates -----
    all_dates = extract_all_dates_in_text(combined)
    seen_dates = set()
    for orig, d in all_dates:
        date_key = str(d)
        if date_key not in seen_dates:
            seen_dates.add(date_key)
            dow = DAY_NAMES[d.weekday()]
            facts.append(f"{d.day} {d.strftime('%B')} {d.year} is a {dow}.")

    # ----- 2. Anchor date + derive other dates in evidence -----
    # Pattern: "1 April 2026 was a Wednesday"
    anchor_match = re.search(
        r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\s+was\s+a\s+(\w+day)',
        evidence, re.IGNORECASE
    )
    if anchor_match:
        try:
            day = int(anchor_match.group(1))
            month = MONTH_MAP[anchor_match.group(2).lower()]
            year = int(anchor_match.group(3))
            anchor_date = date(year, month, day)
            # Cross-check: verify the stated day name matches reality
            stated_dow = anchor_match.group(4).capitalize()
            actual_dow = DAY_NAMES[anchor_date.weekday()]
            if stated_dow.lower() != actual_dow.lower():
                facts.append(f"[NOTE: Evidence says {anchor_match.group(0)} but actual day is {actual_dow}]")
        except (ValueError, KeyError):
            pass

    # ----- 3. Time interval analysis -----
    # Find all closed intervals "from X to Y" or "between X and Y"
    interval_pats = [
        r'(?:from|between)\s+(\d{1,2}:\d{2}\s*(?:AM|PM))\s+(?:to|and)\s+(\d{1,2}:\d{2}\s*(?:AM|PM))',
    ]
    intervals_found = []
    for pat in interval_pats:
        for m in re.finditer(pat, evidence, re.IGNORECASE):
            t1 = parse_time_to_minutes(m.group(1))
            t2 = parse_time_to_minutes(m.group(2))
            if t1 is not None and t2 is not None:
                dur_min = t2 - t1
                dur_hr = dur_min // 60
                dur_rem = dur_min % 60
                duration_str = ""
                if dur_hr > 0:
                    duration_str = f"{dur_hr}h {dur_rem}min" if dur_rem else f"{dur_hr}h"
                else:
                    duration_str = f"{dur_rem}min"
                intervals_found.append((m.group(1).strip(), m.group(2).strip(), t1, t2, duration_str))
                facts.append(
                    f"The interval {m.group(1).strip()} to {m.group(2).strip()} "
                    f"spans {duration_str} and includes both endpoints."
                )

    # ----- 4. Percentage/count arithmetic -----
    # e.g. "Exactly 50 passes were sold and exactly 20% of them were sold before noon"
    pct_match = re.search(r'exactly\s+(\d+)\s+\w+.*?(\d+)%\s+of\s+them', evidence, re.IGNORECASE)
    if pct_match:
        total = int(pct_match.group(1))
        pct = int(pct_match.group(2))
        computed = round(total * pct / 100)
        facts.append(f"{pct}% of {total} = {computed} (exact integer).")

    # ----- 5. Travel time / minimum time constraints -----
    travel_match = re.search(
        r'every route from .+? takes at least (\d+) minutes',
        evidence, re.IGNORECASE
    )
    if travel_match:
        min_travel = int(travel_match.group(1))
        facts.append(f"Minimum travel time specified: {min_travel} minutes.")
    
    # Multi-leg travel (e.g. campus -> bridge -> stadium)
    multi_leg = re.findall(r'every route from (.+?) to (.+?) takes at least (\d+) (\w+)', evidence, re.IGNORECASE)
    if len(multi_leg) >= 2:
        # Sum minimum times
        total_mins = 0
        legs = []
        for leg in multi_leg:
            frm, to, amt, unit = leg
            if 'hour' in unit:
                total_mins += int(amt) * 60
            else:
                total_mins += int(amt)
            legs.append(f"{frm.strip()}->{to.strip()} (≥{amt} {unit})")
        facts.append(f"Minimum total travel time ({' + '.join(legs)}): ≥{total_mins} minutes.")
    
    # ----- 6. Coach/vehicle round-trip -----
    roundtrip = re.search(r'round trip took at least (\d+) hour[s]? (\d+) minutes', evidence, re.IGNORECASE)
    if roundtrip:
        total_min = int(roundtrip.group(1)) * 60 + int(roundtrip.group(2))
        # Find departure time
        dep_match = re.search(r'left .+ at (\d{1,2}:\d{2}\s*(?:AM|PM))', evidence, re.IGNORECASE)
        if dep_match:
            dep = parse_time_to_minutes(dep_match.group(1))
            earliest_return = dep + total_min
            facts.append(
                f"Vehicle departed at {dep_match.group(1).strip()} + ≥{total_min}min round trip => "
                f"earliest return: {minutes_to_hhmm(earliest_return)}."
            )

    # ----- 7. Encode "open only on certain days" -----
    rota_match = re.search(
        r'open only (\w+day)\s+to\s+(\w+day)',
        evidence, re.IGNORECASE
    )
    if rota_match:
        open_days_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        start_day = rota_match.group(1).lower()
        end_day = rota_match.group(2).lower()
        s_idx = open_days_map.get(start_day, -1)
        e_idx = open_days_map.get(end_day, -1)
        if s_idx >= 0 and e_idx >= 0:
            if e_idx >= s_idx:
                open_set = DAY_NAMES[s_idx:e_idx+1]
            else:
                open_set = DAY_NAMES[s_idx:] + DAY_NAMES[:e_idx+1]
            closed_set = [d for d in DAY_NAMES if d not in open_set]
            facts.append(f"Open days ({start_day.capitalize()} to {end_day.capitalize()}): {', '.join(open_set)}.")
            if closed_set:
                facts.append(f"Closed days: {', '.join(closed_set)}.")
            
            # Now check each extracted date against open days
            for orig, d in all_dates:
                dow = DAY_NAMES[d.weekday()]
                if dow in open_set:
                    facts.append(f"=> {d.day} {d.strftime('%B')} {d.year} ({dow}) is an OPEN day.")
                else:
                    facts.append(f"=> {d.day} {d.strftime('%B')} {d.year} ({dow}) is a CLOSED day.")

    if not facts:
        return ""
    
    return "\n[DETERMINISTIC COMPUTED FACTS - These are mathematically verified]\n" + \
           "\n".join(f"  • {f}" for f in facts) + "\n"


if __name__ == '__main__':
    # Test cases
    tests = [
        ("1 April 2026 was a Wednesday. In April 2026 the pool was open only Wednesday to Sunday, from 9:00 AM to 5:00 PM, and nobody could use it while it was closed.",
         "2. D: I swam at the pool at 1:25 PM on 7 April 2026.\n3. A: On 11 April 2026 at 3:15 PM I swam at the pool."),
        ("All times in this case refer to 27 May 2026. The tour coach left the museum gate at 9:55 AM. Its round trip took at least 1 hour 45 minutes, so it could not be back at the museum gate before then.",
         ""),
        ("All times in this case refer to 27 September 2026. Exactly 50 passes were sold for the museum tour, and exactly 20% of them were sold before noon.",
         ""),
        ("All times in this case refer to 3 May 2026. Goran was at the campus at 1:30 PM. The campus, the bridge and the stadium are separate locations. Every route from the campus to the bridge takes at least 10 minutes, while every route from the bridge to the stadium takes at least 30 minutes.",
         ""),
    ]
    
    for evd, chat in tests:
        print("EVIDENCE:", evd[:100])
        result = compute_temporal_facts(evd, chat)
        print(result or "(no computed facts)")
        print()
