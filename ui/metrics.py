"""Metric card rendering and visual event counting."""


def fitness_color_class(pct: float, thresholds: tuple = (65, 40)) -> str:
    """Return CSS class for a fitness percentage."""
    high, mid = thresholds
    if pct >= high:
        return "green"
    if pct >= mid:
        return "amber"
    return "red"


def compute_visual_counts(events_df) -> tuple:
    """Return (vote_count, motion_count) from an events DataFrame."""
    vis_votes = 0
    vis_motion = 0
    if events_df is None or events_df.empty:
        return vis_votes, vis_motion
    if "source" in events_df.columns:
        src = events_df["source"]
        vis_votes = int((src == "Video").sum() + (src == "Fused (Audio+Video)").sum())
        vis_motion = int((src == "Audio+Motion").sum())
        if "activity_name" in events_df.columns:
            vis_motion += int(
                ((events_df["activity_name"] == "Speaker Activity") & (src == "Video")).sum()
            )
    return vis_votes, vis_motion


def render_metrics_html(
    placeholder,
    fitness_raw_pct=0.0, fitness_dedup_pct=0.0,
    events_count=0, agenda_count=0, matched_count=0, shadow_count=0,
    visual_votes=0, visual_motion=0,
):
    """Render the 6-card metric row into a Streamlit placeholder."""
    # Dedup fitness colour
    if events_count == 0:
        dedup_cls = "blue"
    elif fitness_dedup_pct >= 65:
        dedup_cls = "green"
    elif fitness_dedup_pct >= 40:
        dedup_cls = "amber"
    else:
        dedup_cls = "red"

    # Raw fitness colour
    if events_count == 0:
        raw_cls = "blue"
    elif fitness_raw_pct >= 50:
        raw_cls = "green"
    elif fitness_raw_pct >= 25:
        raw_cls = "amber"
    else:
        raw_cls = "red"

    # Visual events colour
    vis_total = visual_votes + visual_motion
    vis_cls = "green" if vis_total > 0 else "blue"

    placeholder.markdown(
        f"""
<div class="metric-row">
    <div class="metric-card {dedup_cls}">
        <div class="label">Dedup Fitness (primary)</div>
        <div class="value">{fitness_dedup_pct:.1f}%</div>
        <div class="sub">First-occurrence trace</div>
    </div>
    <div class="metric-card {raw_cls}">
        <div class="label">Raw Fitness</div>
        <div class="value">{fitness_raw_pct:.1f}%</div>
        <div class="sub">All events (strict)</div>
    </div>
    <div class="metric-card blue">
        <div class="label">Events Detected</div>
        <div class="value">{events_count}</div>
        <div class="sub">Audio + Visual fused</div>
    </div>
    <div class="metric-card purple">
        <div class="label">Agenda Items</div>
        <div class="value">{agenda_count}</div>
        <div class="sub">Reference model tasks</div>
    </div>
    <div class="metric-card amber">
        <div class="label">Matched / Shadow</div>
        <div class="value">{matched_count} / {shadow_count}</div>
        <div class="sub">Formal vs informal</div>
    </div>
    <div class="metric-card {vis_cls}">
        <div class="label">Visual Events</div>
        <div class="value">{vis_total}</div>
        <div class="sub">{visual_votes} votes · {visual_motion} motion</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )
