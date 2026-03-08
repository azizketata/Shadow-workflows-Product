"""Classify and analyze procedural violations from Declare conformance results.

Provides severity classification, violation grouping, and markdown report
generation for presenting compliance findings.
"""

import sys
import os
from collections import Counter, defaultdict
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Violation severity classification
# ---------------------------------------------------------------------------
# Maps Declare template types to severity levels.
# Severity reflects how critical the procedural rule is:
#   high   - Fundamental rules; violation invalidates proceedings
#   medium - Important rules; violation is a procedural defect
#   low    - Best-practice rules; violation is a minor irregularity

VIOLATION_SEVERITY = {
    "existence":        "high",     # Required activities missing entirely
    "absence":          "high",     # Forbidden activity occurred
    "succession":       "high",     # Core ordering + co-occurrence broken
    "response":         "medium",   # Something triggered but never completed
    "precedence":       "medium",   # Order was wrong
    "chain_response":   "low",      # Immediate succession not met (strictest)
    "chain_precedence": "low",      # Immediate predecessor not met
    "not_co_existence": "medium",   # Mutually exclusive activities co-occurred
}

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def classify_violations(
    conformance_results: list,
    constraints: list = None,
) -> list:
    """Classify violated constraints by severity.

    Parameters
    ----------
    conformance_results : list[dict]
        Output from ``check_roberts_rules_conformance()``. Each dict has
        ``template``, ``activity_a``, ``activity_b``, ``description``,
        ``satisfied``, ``detail``.
    constraints : list[tuple], optional
        Original constraint tuples for additional description lookup.
        Not required if conformance_results already contain descriptions.

    Returns
    -------
    list[dict]
        Violated constraints enriched with:
        - ``severity``: "high", "medium", or "low".
        - ``severity_rank``: Numeric rank (0=high, 1=medium, 2=low).
        - ``category``: Human-readable violation category.
        - ``remediation``: Suggested corrective action.
        Sorted by severity (high first).
    """
    if not conformance_results:
        return []

    violations = []
    for result in conformance_results:
        if result.get("satisfied", True):
            continue

        template = result["template"]
        severity = VIOLATION_SEVERITY.get(template, "medium")

        violation = {
            "template": template,
            "activity_a": result.get("activity_a", ""),
            "activity_b": result.get("activity_b"),
            "description": result.get("description", ""),
            "detail": result.get("detail", ""),
            "severity": severity,
            "severity_rank": SEVERITY_ORDER.get(severity, 1),
            "category": _get_violation_category(template, result),
            "remediation": _get_remediation(template, result),
        }
        violations.append(violation)

    # Sort by severity (high first), then by template name
    violations.sort(key=lambda v: (v["severity_rank"], v["template"]))
    return violations


def _get_violation_category(template: str, result: dict) -> str:
    """Assign a human-readable category to a violation."""
    act_a = result.get("activity_a", "")
    act_b = result.get("activity_b", "")

    # Opening/closing ceremony violations
    opening_activities = {"call to order", "roll call", "pledge of allegiance"}
    closing_activities = {"adjourn meeting", "adjournment"}

    a_lower = act_a.lower()
    b_lower = (act_b or "").lower()

    if a_lower in opening_activities or b_lower in opening_activities:
        return "Opening Procedure Violation"
    if a_lower in closing_activities or b_lower in closing_activities:
        return "Closing Procedure Violation"
    if "motion" in a_lower or "motion" in b_lower:
        return "Motion Procedure Violation"
    if "vote" in a_lower or "vote" in b_lower:
        return "Voting Procedure Violation"
    if "hearing" in a_lower or "hearing" in b_lower:
        return "Public Hearing Violation"
    if "consent" in a_lower or "consent" in b_lower:
        return "Consent Agenda Violation"
    if "agenda" in a_lower or "agenda" in b_lower:
        return "Agenda Procedure Violation"
    if "minutes" in a_lower or "minutes" in b_lower:
        return "Minutes Procedure Violation"

    return "General Procedural Violation"


def _get_remediation(template: str, result: dict) -> str:
    """Suggest a corrective action for the violation."""
    act_a = result.get("activity_a", "")
    act_b = result.get("activity_b", "")

    if template == "existence":
        return (
            f"The activity '{act_a}' must be added to the meeting proceedings. "
            f"This is a required procedural step under Robert's Rules."
        )
    elif template == "absence":
        return (
            f"The activity '{act_a}' should not have occurred. "
            f"Review whether this was an unauthorized action."
        )
    elif template == "precedence":
        return (
            f"Ensure '{act_a}' is performed before '{act_b}'. "
            f"The chair should enforce the proper ordering of proceedings."
        )
    elif template == "response":
        return (
            f"After '{act_a}', ensure '{act_b}' is completed before the meeting ends. "
            f"Consider adding a follow-up item to the next meeting agenda if missed."
        )
    elif template == "succession":
        return (
            f"Both '{act_a}' and '{act_b}' must occur, with '{act_a}' first. "
            f"The chair should verify both steps are completed in order."
        )
    elif template == "chain_response":
        return (
            f"After '{act_a}', '{act_b}' should occur immediately without "
            f"intervening activities. The chair should streamline the transition."
        )
    elif template == "chain_precedence":
        return (
            f"'{act_b}' should be immediately preceded by '{act_a}'. "
            f"Remove intervening activities between these steps."
        )
    elif template == "not_co_existence":
        return (
            f"'{act_a}' and '{act_b}' should not both occur in the same meeting. "
            f"Review whether one of these was performed in error."
        )
    else:
        return "Review the procedural requirements and ensure compliance."


def get_violation_summary(violations: list) -> dict:
    """Summarize violations by severity and category.

    Parameters
    ----------
    violations : list[dict]
        Output from ``classify_violations()``.

    Returns
    -------
    dict
        - ``total``: Total violation count.
        - ``by_severity``: Counter of high/medium/low.
        - ``by_category``: Counter of category names.
        - ``most_severe``: The single most severe violation (or None).
        - ``critical_count``: Number of high-severity violations.
    """
    if not violations:
        return {
            "total": 0,
            "by_severity": {},
            "by_category": {},
            "most_severe": None,
            "critical_count": 0,
        }

    by_severity = Counter(v["severity"] for v in violations)
    by_category = Counter(v["category"] for v in violations)

    return {
        "total": len(violations),
        "by_severity": dict(by_severity),
        "by_category": dict(by_category),
        "most_severe": violations[0] if violations else None,
        "critical_count": by_severity.get("high", 0),
    }


def generate_procedural_report(
    violations: list,
    compliance_score: dict,
    meeting_name: str = "Meeting",
) -> str:
    """Generate a markdown report of procedural compliance findings.

    Parameters
    ----------
    violations : list[dict]
        Output from ``classify_violations()``.
    compliance_score : dict
        Output from ``compute_procedural_compliance_score()``.
    meeting_name : str
        Name/identifier of the meeting for the report header.

    Returns
    -------
    str
        Markdown-formatted report string.
    """
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    score = compliance_score.get("score", 0.0)
    grade = compliance_score.get("grade", "N/A")

    # --- Header ---
    lines.append(f"# Procedural Compliance Report: {meeting_name}")
    lines.append(f"*Generated: {timestamp}*")
    lines.append("")

    # --- Score Summary ---
    lines.append("## Compliance Score")
    lines.append("")
    lines.append(f"**Overall Score: {score}/100 (Grade: {grade})**")
    lines.append("")
    lines.append(
        f"- Constraints checked: {compliance_score.get('total_checked', 0)}"
    )
    lines.append(
        f"- Satisfied: {compliance_score.get('satisfied_count', 0)}"
    )
    lines.append(
        f"- Violated: {compliance_score.get('violated_count', 0)}"
    )
    lines.append("")

    # --- Per-template breakdown ---
    by_template = compliance_score.get("by_template", {})
    if by_template:
        lines.append("### Compliance by Rule Type")
        lines.append("")
        lines.append("| Template | Satisfied | Total | Rate |")
        lines.append("|----------|-----------|-------|------|")
        for template, stats in sorted(by_template.items()):
            rate_pct = f"{stats['rate'] * 100:.0f}%"
            lines.append(
                f"| {template} | {stats['satisfied']} | "
                f"{stats['total']} | {rate_pct} |"
            )
        lines.append("")

    # --- Violations ---
    if not violations:
        lines.append("## Violations")
        lines.append("")
        lines.append("No procedural violations detected. The meeting follows Robert's Rules of Order.")
        lines.append("")
    else:
        summary = get_violation_summary(violations)
        lines.append("## Violations Summary")
        lines.append("")
        lines.append(f"**{summary['total']} violation(s) detected:**")
        for sev, count in sorted(
            summary["by_severity"].items(),
            key=lambda x: SEVERITY_ORDER.get(x[0], 99),
        ):
            icon = {"high": "[!]", "medium": "[~]", "low": "[.]"}.get(sev, "[-]")
            lines.append(f"- {icon} {sev.upper()}: {count}")
        lines.append("")

        # Group violations by severity
        for severity_level in ["high", "medium", "low"]:
            level_violations = [
                v for v in violations if v["severity"] == severity_level
            ]
            if not level_violations:
                continue

            severity_label = {
                "high": "Critical Violations",
                "medium": "Moderate Violations",
                "low": "Minor Violations",
            }.get(severity_level, "Violations")

            lines.append(f"### {severity_label}")
            lines.append("")

            for i, v in enumerate(level_violations, 1):
                lines.append(
                    f"**{i}. {v['category']}** ({v['template']})"
                )
                lines.append(f"  - Rule: {v['description']}")
                lines.append(f"  - Finding: {v['detail']}")
                lines.append(f"  - Remediation: {v['remediation']}")
                lines.append("")

    # --- Recommendations ---
    lines.append("## Recommendations")
    lines.append("")

    if score >= 90:
        lines.append(
            "The meeting demonstrates strong procedural compliance. "
            "Continue current practices."
        )
    elif score >= 70:
        lines.append(
            "The meeting shows adequate compliance with room for improvement. "
            "Focus on the violated constraints listed above."
        )
    elif score >= 50:
        lines.append(
            "Significant procedural gaps were identified. The chair should "
            "review Robert's Rules of Order and ensure all required steps "
            "are followed in future meetings."
        )
    else:
        lines.append(
            "The meeting shows substantial non-compliance with Robert's Rules. "
            "Consider providing Robert's Rules training for the chair and "
            "council members. A parliamentarian may be needed for future "
            "meetings to ensure proper procedure."
        )
    lines.append("")

    return "\n".join(lines)
