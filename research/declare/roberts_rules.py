"""Robert's Rules of Order formalized as Declare temporal constraints.

Each constraint is a tuple:
    (template, activity_A, activity_B, description)

Templates used (pm4py Declare templates):
    - precedence:       B can only occur if A has occurred before
    - response:         If A occurs, B must eventually occur after
    - succession:       A and B must both occur, A before B (precedence + response)
    - chain_response:   If A occurs, B must occur immediately next
    - chain_precedence: B can only occur if A occurred immediately before
    - not_co_existence: A and B cannot both occur in the same trace
    - existence:        A must occur at least once (unary)
    - absence:          A must not occur (unary, used for negative constraints)

The constraints below encode the core procedural requirements of Robert's
Rules of Order as practiced in U.S. city council meetings.
"""


# ---------------------------------------------------------------------------
# Normative Declare constraints derived from Robert's Rules of Order
# ---------------------------------------------------------------------------
# Format: (template, activity_A, activity_B, description)
# For unary templates (existence, absence), activity_B is None.
# ---------------------------------------------------------------------------

ROBERTS_RULES_CONSTRAINTS = [
    # === Meeting Opening Sequence ===
    (
        "existence",
        "Call to Order",
        None,
        "Every meeting must be formally called to order (Robert's Rules Ch. I).",
    ),
    (
        "precedence",
        "Call to Order",
        "Roll Call",
        "Roll call can only occur after the meeting is called to order.",
    ),
    (
        "precedence",
        "Call to Order",
        "Approval of Agenda",
        "The agenda cannot be approved until the meeting is called to order.",
    ),
    (
        "precedence",
        "Roll Call",
        "Approval of Agenda",
        "The agenda is approved after roll call confirms a quorum is present.",
    ),
    (
        "response",
        "Call to Order",
        "Adjourn Meeting",
        "A meeting that is called to order must eventually be adjourned.",
    ),

    # === Consent Agenda ===
    (
        "precedence",
        "Approval of Agenda",
        "Approve Consent Agenda Items",
        "Consent items are handled after the agenda is formally approved.",
    ),
    (
        "precedence",
        "Approval of Minutes",
        "Approve Consent Agenda Items",
        "Previous meeting minutes must be approved before new consent business.",
    ),

    # === Motion Procedure (Robert's Rules Ch. IV-VI) ===
    (
        "response",
        "Propose Motion",
        "Call for Vote",
        "Every motion must eventually receive a vote (or be withdrawn).",
    ),
    (
        "precedence",
        "Propose Motion",
        "Second Motion",
        "A motion can only be seconded after it has been proposed.",
    ),
    (
        "precedence",
        "Second Motion",
        "Call for Vote",
        "A vote can only be called after the motion has been seconded.",
    ),
    (
        "chain_response",
        "Propose Motion",
        "Second Motion",
        "A motion should be immediately followed by a request for a second.",
    ),

    # === Public Hearings (required for ordinances in many jurisdictions) ===
    (
        "response",
        "Open Public Hearing",
        "Close Public Hearing",
        "A public hearing that is opened must eventually be closed.",
    ),
    (
        "precedence",
        "Open Public Hearing",
        "Close Public Hearing",
        "A public hearing cannot be closed unless it was first opened.",
    ),
    (
        "precedence",
        "Close Public Hearing",
        "Call for Vote",
        "Voting on a hearing item can only occur after the hearing is closed.",
    ),

    # === Meeting Closing ===
    (
        "existence",
        "Adjourn Meeting",
        None,
        "Every meeting must be formally adjourned (Robert's Rules Ch. XXI).",
    ),
]


# ---------------------------------------------------------------------------
# Helper: canonical activity names used in the constraints above
# ---------------------------------------------------------------------------

CANONICAL_ACTIVITIES = sorted(set(
    c[1] for c in ROBERTS_RULES_CONSTRAINTS
) | set(
    c[2] for c in ROBERTS_RULES_CONSTRAINTS if c[2] is not None
))


def get_constraints_for_agenda(agenda_activities: list) -> list:
    """Filter constraints to only include activities present in the agenda.

    Parameters
    ----------
    agenda_activities : list[str]
        The agenda activity labels extracted for this specific meeting.

    Returns
    -------
    list[tuple]
        Subset of ``ROBERTS_RULES_CONSTRAINTS`` where both activity_A and
        activity_B (if not None) appear in ``agenda_activities``.
    """
    if not agenda_activities:
        return []

    # Normalize to lowercase set for matching
    agenda_lower = {a.lower().strip() for a in agenda_activities}

    filtered = []
    for template, act_a, act_b, desc in ROBERTS_RULES_CONSTRAINTS:
        a_present = act_a.lower().strip() in agenda_lower
        b_present = act_b is None or act_b.lower().strip() in agenda_lower

        if a_present and b_present:
            filtered.append((template, act_a, act_b, desc))

    return filtered


def constraints_to_pm4py_format(constraints: list) -> dict:
    """Convert constraint list to pm4py ``conformance_declare`` format.

    pm4py's ``conformance_declare`` expects a dict of the form::

        {
            "existence": {"Activity": {"min": 1}},
            "precedence": [("A", "B")],
            "response": [("A", "B")],
            ...
        }

    This function builds that structure from the constraint tuples.

    Parameters
    ----------
    constraints : list[tuple]
        List of ``(template, activity_A, activity_B, description)`` tuples.

    Returns
    -------
    dict
        Dictionary suitable for pm4py Declare conformance checking.
    """
    # Collect constraints by template type
    model = {}

    for template, act_a, act_b, _desc in constraints:
        if template == "existence":
            # Unary: activity must occur at least once
            if "existence" not in model:
                model["existence"] = {}
            model["existence"][act_a] = {"min": 1}

        elif template == "absence":
            # Unary: activity must not occur
            if "absence" not in model:
                model["absence"] = {}
            model["absence"][act_a] = {"max": 0}

        else:
            # Binary templates: add as (A, B) pair
            if template not in model:
                model[template] = []
            model[template].append((act_a, act_b))

    return model


def get_constraint_descriptions() -> dict:
    """Return a mapping from (template, A, B) to human-readable description.

    Returns
    -------
    dict
        Key is ``(template, activity_A, activity_B)``; value is the
        description string.
    """
    return {
        (c[0], c[1], c[2]): c[3]
        for c in ROBERTS_RULES_CONSTRAINTS
    }
