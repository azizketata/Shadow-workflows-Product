"""Shareable Deep Links — generate and parse URL query params for
jumping to a specific moment or agenda item in the Meeting Process Twin."""

import streamlit as st
import urllib.parse


def apply_deep_link_params() -> tuple:
    """Read st.query_params on page load and return navigation targets.

    Returns:
        (target_time, target_item) where target_time is seconds (int)
        and target_item is an agenda-item label (str) or None.
    """
    target_time = 0
    target_item = None

    raw_t = st.query_params.get("t", None)
    if raw_t is not None:
        try:
            target_time = int(raw_t)
        except (ValueError, TypeError):
            target_time = 0

    raw_item = st.query_params.get("item", None)
    if raw_item is not None:
        target_item = str(raw_item)

    return target_time, target_item


def generate_deep_link(seconds: int, item: str = None) -> str:
    """Generate a URL query string for a specific moment.

    Args:
        seconds: Timestamp in seconds to link to.
        item:    Optional agenda-item label.

    Returns:
        Query string like ``?t=1415`` or ``?t=1415&item=Public+Comment``.
    """
    query = f"?t={seconds}"
    if item is not None:
        query += f"&item={urllib.parse.quote(item)}"
    return query


def render_share_button(seconds: int, item: str = None):
    """Display a copyable deep-link inside an ``st.code`` block."""
    link = generate_deep_link(seconds, item)
    label = "Link to this moment"
    if item:
        label += f" ({item})"
    st.caption(label)
    st.code(link, language=None)
