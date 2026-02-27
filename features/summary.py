"""AI Meeting Summary — one-shot LLM narrative from mapped events."""

import hashlib
import json
import pandas as pd
import openai
import streamlit as st


def _events_hash(mapped_events: pd.DataFrame) -> str:
    """Stable hash of mapped events for session-state caching."""
    raw = mapped_events.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def extract_vote_events(mapped_events: pd.DataFrame) -> list[dict]:
    """Find vote-related events by activity name or fused audio+video source.

    Returns list of {"timestamp": str, "text": str, "hands_count": int}.
    """
    if mapped_events.empty:
        return []

    vote_keywords = ["vote", "motion", "confirmed vote"]
    fused_sources = ["fused (audio+video)", "fused"]
    results = []
    for _, row in mapped_events.iterrows():
        activity = str(row.get("activity_name", "")).lower()
        source = str(row.get("source", "")).lower()
        is_vote = any(kw in activity for kw in vote_keywords)
        is_fused = any(fs in source for fs in fused_sources)
        if is_vote or (is_fused and "confirmed vote" in activity):
            results.append({
                "timestamp": str(row.get("timestamp", "")),
                "text": str(row.get("original_text", row.get("details", ""))),
                "hands_count": int(row.get("hands_count", 0)),
            })
    return results


def _build_summary_prompt(
    mapped_events: pd.DataFrame, agenda_activities: list, votes: list[dict],
) -> str:
    """Build the single-call LLM prompt."""
    lines = []
    for _, row in mapped_events.iterrows():
        ts = row.get("timestamp", "")
        label = row.get("mapped_activity", row.get("activity_name", ""))
        snippet = str(row.get("original_text", ""))[:120]
        lines.append(f"[{ts}] {label} — {snippet}")
    event_stream = "\n".join(lines[:200])  # cap to avoid token overflow

    agenda_str = "\n".join(f"  - {a}" for a in agenda_activities) if agenda_activities else "(none)"

    vote_str = ""
    if votes:
        vote_lines = [f"  [{v['timestamp']}] {v['text'][:100]} (hands: {v['hands_count']})" for v in votes]
        vote_str = "\nDETECTED VOTES:\n" + "\n".join(vote_lines)

    return (
        "You are a professional city-council meeting analyst. "
        "Given the chronological event log below, produce a structured meeting summary.\n\n"
        f"AGENDA ITEMS:\n{agenda_str}\n\n"
        f"CHRONOLOGICAL EVENT LOG:\n{event_stream}\n{vote_str}\n\n"
        "Return a JSON object (no markdown fences) with exactly these keys:\n"
        '  "summary": a 3-5 sentence narrative overview of the meeting,\n'
        '  "key_decisions": list of strings — decisions made,\n'
        '  "action_items": list of strings — follow-up tasks identified,\n'
        '  "votes": list of strings — description of each vote (include result if known)\n\n'
        "If no items exist for a key, use an empty list. Return ONLY valid JSON."
    )


def generate_meeting_summary(
    mapped_events: pd.DataFrame,
    agenda_activities: list,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Make one OpenAI call to produce a structured meeting summary.

    Returns {"summary": str, "key_decisions": [str], "action_items": [str], "votes": [str]}.
    """
    empty = {"summary": "", "key_decisions": [], "action_items": [], "votes": []}
    if mapped_events is None or mapped_events.empty:
        return empty

    votes = extract_vote_events(mapped_events)
    prompt = _build_summary_prompt(mapped_events, agenda_activities, votes)

    try:
        client = openai.OpenAI(api_key=api_key, timeout=90)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise meeting analyst. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if the model wraps them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
        return {
            "summary": result.get("summary", ""),
            "key_decisions": result.get("key_decisions", []),
            "action_items": result.get("action_items", []),
            "votes": result.get("votes", []),
        }
    except Exception as e:
        return {**empty, "summary": f"Summary generation failed: {e}"}


def render_summary(summary_dict: dict) -> None:
    """Streamlit UI: narrative summary, key decisions, action items, votes."""
    if not summary_dict or not summary_dict.get("summary"):
        st.info("No meeting summary available. Generate one using the button above.")
        return

    st.subheader("Meeting Summary")
    st.markdown(summary_dict["summary"])

    with st.expander("Key Decisions", expanded=True):
        decisions = summary_dict.get("key_decisions", [])
        if decisions:
            for d in decisions:
                st.markdown(f"- {d}")
        else:
            st.write("No key decisions recorded.")

    with st.expander("Action Items", expanded=True):
        items = summary_dict.get("action_items", [])
        if items:
            for item in items:
                st.markdown(f"- {item}")
        else:
            st.write("No action items identified.")

    with st.expander("Votes", expanded=False):
        votes = summary_dict.get("votes", [])
        if votes:
            for v in votes:
                st.markdown(f"- {v}")
        else:
            st.write("No votes detected.")


def get_or_generate_summary(
    mapped_events: pd.DataFrame,
    agenda_activities: list,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Cache-aware wrapper — skips LLM call if events haven't changed.

    Caches in st.session_state by SHA-256 of the events DataFrame,
    so slider moves that don't alter data won't re-trigger the API call.
    """
    cache_key = "summary_" + _events_hash(mapped_events)
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    result = generate_meeting_summary(mapped_events, agenda_activities, api_key, model)
    st.session_state[cache_key] = result
    return result
