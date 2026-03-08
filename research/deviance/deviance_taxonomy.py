"""Taxonomy of process deviance types in civic meetings.

Classifies shadow workflows into categories based on their nature and impact:
- BENIGN_FLEXIBILITY: Normal adaptive behavior (breaks, informal greetings)
- PROCEDURAL_VIOLATION: Robert's Rules violations (voting without motion)
- EFFICIENCY_GAIN: Streamlining behavior (combining agenda items)
- INNOVATION: Novel deliberation approaches
- EXTERNAL_DISRUPTION: Technical issues, interruptions
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class DevianceCategory(Enum):
    """Categories of process deviance observed in civic meetings."""
    BENIGN_FLEXIBILITY = "benign"
    PROCEDURAL_VIOLATION = "violation"
    EFFICIENCY_GAIN = "efficient"
    INNOVATION = "innovation"
    EXTERNAL_DISRUPTION = "disruption"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Rule-based classification patterns
# Each key is a regex pattern matched against the combined activity_name + text.
# The value is the DevianceCategory assigned when the pattern matches.
# Patterns are checked in order; first match wins.
# ---------------------------------------------------------------------------
DEVIANCE_RULES: dict[str, DevianceCategory] = {
    # Benign flexibility -- normal adaptive behavior
    r"break|recess|intermission": DevianceCategory.BENIGN_FLEXIBILITY,
    r"thank|greeting|welcome|birthday|wish": DevianceCategory.BENIGN_FLEXIBILITY,
    r"sidebar|off.?record|side.?conversation": DevianceCategory.BENIGN_FLEXIBILITY,
    r"small.?talk|chit.?chat|informal.?(conversation|discussion|remark|exchange)": DevianceCategory.BENIGN_FLEXIBILITY,
    r"informal.?(discussion|remark|side|comment|chat|banter)": DevianceCategory.BENIGN_FLEXIBILITY,
    r"water|coffee|restroom": DevianceCategory.BENIGN_FLEXIBILITY,
    r"pre.?meeting|post.?meeting|before.*start": DevianceCategory.BENIGN_FLEXIBILITY,
    r"personal.*remark|joke|laughter": DevianceCategory.BENIGN_FLEXIBILITY,

    # Procedural violations -- Robert's Rules violations
    r"vote.*without.*motion|motion.*without.*second": DevianceCategory.PROCEDURAL_VIOLATION,
    r"out.*of.*order|point.*of.*order": DevianceCategory.PROCEDURAL_VIOLATION,
    r"speak.*out.*of.*turn|interrupt": DevianceCategory.PROCEDURAL_VIOLATION,
    r"no.*quorum|quorum.*not.*met": DevianceCategory.PROCEDURAL_VIOLATION,
    r"debate.*after.*vote|vote.*before.*discussion": DevianceCategory.PROCEDURAL_VIOLATION,
    r"chair.*ruling.*overturned|overrule": DevianceCategory.PROCEDURAL_VIOLATION,
    r"motion.*not.*recognized|unrecognized.*motion": DevianceCategory.PROCEDURAL_VIOLATION,
    r"no.*second|without.*second": DevianceCategory.PROCEDURAL_VIOLATION,

    # Efficiency gains -- streamlining behavior
    r"skip|combine|streamline": DevianceCategory.EFFICIENCY_GAIN,
    r"consent.*bundle|batch.*approval": DevianceCategory.EFFICIENCY_GAIN,
    r"expedite|fast.?track": DevianceCategory.EFFICIENCY_GAIN,
    r"waive.*reading|dispense.*reading": DevianceCategory.EFFICIENCY_GAIN,
    r"table.*for.*later|defer|postpone": DevianceCategory.EFFICIENCY_GAIN,
    r"move.*previous.*question|call.*question": DevianceCategory.EFFICIENCY_GAIN,

    # Innovation -- novel deliberation approaches
    r"workshop|brainstorm|roundtable": DevianceCategory.INNOVATION,
    r"straw.?poll|informal.*vote|sense.*of.*body": DevianceCategory.INNOVATION,
    r"committee.*of.*whole|work.*session": DevianceCategory.INNOVATION,
    r"presentation.*from.*public|community.*input.*session": DevianceCategory.INNOVATION,
    r"digital.*vote|electronic.*poll": DevianceCategory.INNOVATION,

    # External disruptions -- technical issues, interruptions
    # NOTE: These must come BEFORE catch-all shadow patterns below so that
    # "Shadow: discussion about microphone issues" matches disruption, not innovation.
    r"technical.*difficult|audio.*issue|connection.*issue": DevianceCategory.EXTERNAL_DISRUPTION,
    r"power.*outage|mic.*not.*working|microphone": DevianceCategory.EXTERNAL_DISRUPTION,
    r"zoom.*issue|video.*issue|stream.*issue": DevianceCategory.EXTERNAL_DISRUPTION,
    r"fire.*alarm|emergency(?!.*response)|evacuat": DevianceCategory.EXTERNAL_DISRUPTION,
    r"protester|outburst|removed.*from": DevianceCategory.EXTERNAL_DISRUPTION,
    r"granicus|technical.*glitch": DevianceCategory.EXTERNAL_DISRUPTION,

    # Catch-all shadow patterns -- off-agenda discussions are the dominant shadow type
    # These come LAST so more specific patterns above take priority.
    r"shadow:.*announcement": DevianceCategory.BENIGN_FLEXIBILITY,
    r"shadow:.*recognition|shadow:.*tribute|shadow:.*honor|shadow:.*proclamation": DevianceCategory.BENIGN_FLEXIBILITY,
    r"shadow:.*appreciation|shadow:.*gratitude|shadow:.*congratulat": DevianceCategory.BENIGN_FLEXIBILITY,
    r"shadow:.*moment.*of.*silence|shadow:.*personal.*story": DevianceCategory.BENIGN_FLEXIBILITY,
    r"shadow:.*introduction.*of|shadow:.*mention.*of.*guest": DevianceCategory.BENIGN_FLEXIBILITY,
    r"shadow:.*expression.*of": DevianceCategory.BENIGN_FLEXIBILITY,
    r"unscheduled.*discussion": DevianceCategory.INNOVATION,
    r"off.?agenda.*discussion": DevianceCategory.INNOVATION,
    r"off.?agenda.*debate": DevianceCategory.INNOVATION,
    r"off.?agenda.*frustration|off.?agenda.*summary": DevianceCategory.INNOVATION,
    r"shadow:.*discussion": DevianceCategory.INNOVATION,
    r"shadow:.*update|shadow:.*report": DevianceCategory.INNOVATION,
    r"shadow:.*debate": DevianceCategory.INNOVATION,
}

# Compiled regex cache for performance
_COMPILED_RULES: list[tuple[re.Pattern, DevianceCategory]] | None = None


def _get_compiled_rules() -> list[tuple[re.Pattern, DevianceCategory]]:
    """Lazily compile and cache all deviance rule patterns."""
    global _COMPILED_RULES
    if _COMPILED_RULES is None:
        _COMPILED_RULES = [
            (re.compile(pattern, re.IGNORECASE), category)
            for pattern, category in DEVIANCE_RULES.items()
        ]
    return _COMPILED_RULES


# ---------------------------------------------------------------------------
# Severity weights -- higher = more concerning for governance
# ---------------------------------------------------------------------------
SEVERITY_WEIGHTS: dict[DevianceCategory, int] = {
    DevianceCategory.PROCEDURAL_VIOLATION: 3,
    DevianceCategory.EXTERNAL_DISRUPTION: 2,
    DevianceCategory.UNKNOWN: 1,
    DevianceCategory.BENIGN_FLEXIBILITY: 0,
    DevianceCategory.EFFICIENCY_GAIN: 0,
    DevianceCategory.INNOVATION: 0,
}


# ---------------------------------------------------------------------------
# Human-readable descriptions for each category
# ---------------------------------------------------------------------------
CATEGORY_DESCRIPTIONS: dict[DevianceCategory, str] = {
    DevianceCategory.BENIGN_FLEXIBILITY: (
        "Normal adaptive behavior such as breaks, informal greetings, and "
        "social interaction that does not affect meeting outcomes."
    ),
    DevianceCategory.PROCEDURAL_VIOLATION: (
        "Violation of Robert's Rules of Order or established parliamentary "
        "procedure, potentially affecting the legitimacy of decisions."
    ),
    DevianceCategory.EFFICIENCY_GAIN: (
        "Streamlining behavior that deviates from strict procedure but "
        "improves meeting efficiency (e.g., combining items, waiving readings)."
    ),
    DevianceCategory.INNOVATION: (
        "Novel deliberation approaches that go beyond standard parliamentary "
        "procedure (e.g., workshops, straw polls, community input sessions)."
    ),
    DevianceCategory.EXTERNAL_DISRUPTION: (
        "External factors disrupting the meeting process, including technical "
        "failures, emergencies, or audience disruptions."
    ),
    DevianceCategory.UNKNOWN: (
        "Deviance that does not clearly fit any established category and "
        "requires further analysis or LLM-based classification."
    ),
}


def classify_by_rules(activity_name: str, text: str) -> DevianceCategory:
    """Classify a shadow event using keyword rules.

    Combines the activity_name and text into a single search string, then
    checks each rule pattern in order. Returns the first matching category
    or UNKNOWN if no rule matches.

    Args:
        activity_name: The activity label (e.g. "Shadow: Side conversation
            about parking enforcement").
        text: Additional text context -- typically the 'details' or
            'original_text' field from the event DataFrame.

    Returns:
        The matching DevianceCategory, or DevianceCategory.UNKNOWN.
    """
    combined = f"{activity_name} {text}".strip()
    if not combined:
        return DevianceCategory.UNKNOWN

    for pattern, category in _get_compiled_rules():
        if pattern.search(combined):
            return category

    return DevianceCategory.UNKNOWN


def get_severity(category: DevianceCategory) -> int:
    """Return the severity weight for a deviance category.

    Args:
        category: A DevianceCategory enum member.

    Returns:
        Integer severity (0 = benign, 3 = most concerning).
    """
    return SEVERITY_WEIGHTS.get(category, 1)


def describe_category(category: DevianceCategory) -> str:
    """Return a human-readable description of a deviance category.

    Args:
        category: A DevianceCategory enum member.

    Returns:
        Multi-sentence description string.
    """
    return CATEGORY_DESCRIPTIONS.get(category, "No description available.")
