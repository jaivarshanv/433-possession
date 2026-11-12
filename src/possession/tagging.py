"""Smart Tagging as a tidy pandas DataFrame.

Wraps the official loader (official/starter/load_smart_tagging.py), which already converts the
SportsCode XML to video frames (frame = round(seconds * 25)), and adds:

    code         upper-case, stripped ("SET PIECES")
    team         "H", "A" or None  (36 instances are tagged N/A, including EVERY throw-in)
    half         1 or 2
    type         set-piece type in lower case ("throw-in", "corner kick", ...) or None
    start_frame  first frame of the phase
    end_frame    first frame AFTER the phase (half-open, like the official loader)

Remember: these are automatic, unverified PHASES, not possessions. Use them as soft evidence.
"""

import pandas as pd

from . import config


def team_code(name):
    """Map a club name from the XML to our side label. The XML spells Athletic as 'Atletic'."""
    if not name:
        return None
    lowered = name.strip().lower()
    if "barcelona" in lowered:
        return "H"
    if "atletic" in lowered or "athletic" in lowered:
        return "A"
    return None  # "N/A" and anything unexpected


def load_tagging():
    """Return the Smart Tagging instances as a DataFrame, sorted by start frame."""
    config.add_official_to_path()
    import load_smart_tagging  # official module, needs the path above

    rows = []
    for event in load_smart_tagging.load():
        tags = event["tags"]
        half_text = (tags.get("Half") or "").lower()
        rows.append({
            "code": (event["code"] or "").strip().upper(),
            "team": team_code(tags.get("Team")),
            "half": 1 if half_text.startswith("1") else (2 if half_text.startswith("2") else None),
            "type": (tags.get("Type") or "").strip().lower() or None,
            "start_frame": event["start_frame"],
            "end_frame": event["end_frame"],
            "tags": tags,
        })
    return pd.DataFrame(rows).sort_values("start_frame").reset_index(drop=True)
