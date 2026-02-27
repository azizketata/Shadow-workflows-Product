"""Timestamp conversion utilities used across the pipeline."""

import pandas as pd


def ts_to_seconds(t_str) -> int:
    """Convert 'HH:MM:SS' or 'MM:SS' string to integer seconds.

    Also handles numeric types and NaN gracefully.
    """
    if pd.isna(t_str):
        return 0
    if isinstance(t_str, (int, float)):
        return int(t_str)
    parts = str(t_str).split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        pass
    return 0


def seconds_to_ts(seconds) -> str:
    """Convert numeric seconds to 'HH:MM:SS' string."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def make_time_options(max_seconds: int, step: int = 10) -> list:
    """Generate a list of 'HH:MM:SS' strings from 0 to max_seconds+60."""
    options = [seconds_to_ts(s) for s in range(0, max_seconds + 60, step)]
    return options if options else ["00:00:00"]
