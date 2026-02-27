"""
Shared keyword rules and text utilities for Meeting Process Twin.

Contains:
- Council-specific keyword rules for high-confidence event-to-agenda mapping
- Text condensing utilities for LLM token efficiency
- Visual event summarization for LLM context
"""

import re
from typing import Optional, List
from collections import Counter


# ---------------------------------------------------------------------------
# Keyword rules — high-confidence mapping from NLP labels + text to agenda items
# Each rule: (activity_name_keywords: set, text_patterns: list[regex], agenda_substring: str)
# ---------------------------------------------------------------------------

KEYWORD_RULES = [
    ({"Call to Order", "Call Order"},       [r"come to order", r"call to order", r"meeting.*order"],                "Call to Order"),
    ({"Call Roll", "Roll Call"},            [r"roll call", r"council members present", r"please call roll"],        "Roll Call"),
    ({"Pledge"},                            [r"pledge of allegiance"],                                              "Pledge of Allegiance"),
    ({"Approve Agenda", "Approval Agenda"}, [r"approval of.*agenda", r"approve.*agenda", r"adopt.*agenda"],        "Approval of Agenda"),
    ({"Approve Minutes"},                   [r"approval of.*minutes", r"approve.*minutes", r"previous meeting"],   "Approval of Minutes"),
    ({"Public Comment"},                    [r"public comment", r"members of the public", r"public.*testimony", r"open.*public comment"], "Public Comment (General)"),
    ({"Consent Agenda"},                    [r"consent agenda", r"consent calendar"],                               "Consent Agenda"),
    ({"Staff Report", "Staff Reports"},     [r"staff report", r"city manager report", r"department update", r"presentation"], "Staff Reports and Presentations"),
    ({"City Manager"},                      [r"city manager"],                                                      "City Manager Report"),
    ({"Finance", "Budget"},                 [r"finance department", r"finance update", r"budget update"],           "Finance Department Update"),
    ({"Public Hearing", "Open Hearing"},    [r"open.*public hearing", r"public hearing.*open"],                     "Open Public Hearing"),
    ({"Close Hearing"},                     [r"close.*public hearing", r"public hearing.*close"],                   "Close Public Hearing"),
    ({"Old Business"},                      [r"old business", r"pending.*ordinance"],                               "Old Business"),
    ({"Discuss Ordinance", "Review Ordinance"}, [r"ordinance.*review", r"review.*ordinance", r"pending ordinance"], "Review Pending Ordinances"),
    ({"Budget Amendment", "Discuss Budget"}, [r"budget amendment", r"budget discussion"],                           "Budget Amendments Discussion"),
    ({"New Business"},                      [r"new business"],                                                      "New Business"),
    ({"Propose Motion", "Introduce Motion"}, [r"introduce.*ordinance", r"new ordinance", r"ordinance.*introduce"], "Introduce New Ordinance"),
    ({"Resolution"},                        [r"capital improvement", r"resolution.*capital"],                       "Resolution for Capital Improvements"),
    ({"Contract"},                          [r"contract approval", r"approve.*contract"],                           "Contract Approval"),
    ({"Council Member Report", "Council Reports"}, [r"council member report", r"council.*report"],                  "Council Member Reports"),
    ({"City Attorney"},                     [r"city attorney", r"attorney report"],                                 "City Attorney Report"),
    ({"Scheduling", "Announcements"},       [r"scheduling", r"announcement", r"next meeting"],                      "Scheduling and Announcements"),
    ({"Adjourn", "Adjourned"},              [r"adjourn", r"meeting.*adjourn", r"stand.*recess"],                    "Adjourn Meeting"),
    ({"Call for Vote", "Propose Motion"},   [r"all in favor", r"motion carries", r"ayes have it", r"call.*vote", r"move to approve", r"second.*motion"], "Approve Consent Agenda Items"),
]


def find_activity(activities: list, substr: str) -> Optional[str]:
    """Find the first activity label containing the given substring (case-insensitive)."""
    for a in activities:
        if substr.lower() in a.lower():
            return a
    return None


def keyword_map(row: dict, activities: list) -> Optional[str]:
    """Try to map a row to an agenda item via keyword rules. Returns activity label or None."""
    act_name = str(row.get("activity_name", "")).strip()
    details = str(row.get("details", "")).lower()
    orig_txt = str(row.get("original_text", "")).lower()
    combined_text = (details + " " + orig_txt)[:500]

    for act_keywords, text_patterns, agenda_substr in KEYWORD_RULES:
        # Check NLP activity label
        if act_name in act_keywords:
            if not text_patterns or any(re.search(p, combined_text) for p in text_patterns):
                match = find_activity(activities, agenda_substr)
                if match:
                    return match
        # Check text patterns even if activity_name didn't match
        if text_patterns and any(re.search(p, combined_text) for p in text_patterns):
            match = find_activity(activities, agenda_substr)
            if match:
                return match
    return None


def summarize_visual_events(window_df) -> str:
    """Summarize visual events in a window for LLM context.

    Returns a string like "Visual: 2 hand raises detected, 3 motion spikes"
    or empty string if no visual events.
    """
    if window_df is None or window_df.empty:
        return ""

    source_col = window_df.get("source")
    if source_col is None:
        return ""

    visual_mask = source_col.isin(["Video", "Fused (Audio+Video)", "Audio+Motion"])
    visual = window_df[visual_mask]
    if visual.empty:
        return ""

    act_col = visual.get("activity_name")
    if act_col is None:
        return ""

    hand_raises = int((act_col.str.contains("Voting", na=False) | act_col.str.contains("Confirmed Vote", na=False)).sum())
    motion = int((act_col == "Speaker Activity").sum())

    parts = []
    if hand_raises > 0:
        parts.append(f"{hand_raises} hand raise{'s' if hand_raises != 1 else ''} detected")
    if motion > 0:
        parts.append(f"{motion} motion spike{'s' if motion != 1 else ''}")

    return f"Visual: {', '.join(parts)}" if parts else ""


def condense_window_text(window_df, start_seconds: int, end_seconds: int) -> str:
    """Condense verbose window events into a token-efficient summary.

    Instead of listing every event line, produces:
      Window 00:05:00-00:06:00 (12 events):
        Discussion x5, Public Comment x3, Vote x2, Other x2
        Key text: "city manager presented quarterly budget" | "public comment on zoning"
        Visual: 2 hand raises detected
    """
    def _fmt_ts(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    n_events = len(window_df)

    # Activity name counts
    activity_counts = Counter(window_df["activity_name"].tolist())
    counts_str = ", ".join(f"{name} x{cnt}" for name, cnt in activity_counts.most_common(6))
    if len(activity_counts) > 6:
        counts_str += f", ... +{len(activity_counts) - 6} more"

    # Collect representative text snippets (up to 3, deduped, max 80 chars each)
    seen_snippets = set()
    key_texts = []
    for _, row in window_df.iterrows():
        detail = str(row.get("details", "")).strip()
        if detail and detail != "nan" and len(detail) > 15:
            snippet = detail[:80].strip()
            snippet_key = snippet[:30].lower()
            if snippet_key not in seen_snippets:
                seen_snippets.add(snippet_key)
                key_texts.append(f'"{snippet}"')
        if len(key_texts) >= 3:
            break

    # Visual summary
    visual_note = summarize_visual_events(window_df)

    lines = [
        f"Window {_fmt_ts(start_seconds)}-{_fmt_ts(end_seconds)} ({n_events} events):",
        f"  {counts_str}",
    ]
    if key_texts:
        lines.append(f"  Key text: {' | '.join(key_texts)}")
    if visual_note:
        lines.append(f"  {visual_note}")

    return "\n".join(lines)
