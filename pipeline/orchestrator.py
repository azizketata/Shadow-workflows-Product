"""Pipeline phase functions — flat, explicit inputs/outputs, no session state access."""

import tempfile
import os
import pandas as pd

from video_processor import VideoProcessor
from bpmn_gen import convert_to_event_log
from pipeline.time_utils import ts_to_seconds


def run_video_processing(
    video_file,
    api_key: str,
    use_local_whisper: bool,
    local_whisper_model: str,
    debug: bool,
    progress_callback=None,
    log_fn=None,
) -> pd.DataFrame:
    """Phase 2: Process uploaded video file → raw events DataFrame."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tfile.write(video_file.read())
        video_path = tfile.name
        tfile.close()

        processor = VideoProcessor(api_key or "local", debug=debug)
        df = processor.process_video(
            video_path,
            use_local_whisper=use_local_whisper,
            local_whisper_model=local_whisper_model,
            progress_callback=progress_callback,
        )
        if log_fn:
            log_fn(f"Video processing complete — {df.shape[0]} events")
        return df
    finally:
        if os.path.exists(tfile.name):
            os.unlink(tfile.name)


def run_abstraction(
    events_df: pd.DataFrame,
    agenda_activities: list,
    config: dict,
    cache: dict,
    compliance_engine,
    log_fn=None,
) -> pd.DataFrame:
    """Phase 3: LLM abstraction → abstracted events DataFrame.

    Returns the original events_df unchanged if abstraction is disabled or fails.
    """
    if (
        not config.get("enable_abstraction")
        or events_df.empty
        or not agenda_activities
        or not config.get("api_key")
    ):
        return events_df

    try:
        abstracted = compliance_engine.abstract_events_df(
            df=events_df,
            agenda_tasks=agenda_activities,
            window_seconds=config["window_seconds"],
            shadow_min_ratio=config["shadow_min_ratio"],
            model=config["abstraction_model"],
            overlap_ratio=config["overlap_ratio"],
            min_events_per_window=config["min_events_per_window"],
            min_label_support=config["min_label_support"],
            api_key=config["api_key"],
            openai_model=config["openai_abstraction_model"],
            openai_timeout=config["openai_timeout"],
            max_windows_per_run=config["max_windows_per_run"],
            cache=cache,
            debug_callback=log_fn,
        )
        if log_fn:
            log_fn(f"Abstraction produced {len(abstracted)} events")
        return abstracted
    except Exception as e:
        if log_fn:
            log_fn(f"Abstraction error: {e}")
        return events_df


def run_compliance(
    abstracted_df: pd.DataFrame,
    reference_bpmn,
    agenda_activities: list,
    sbert_threshold: float,
    compliance_engine,
    log_fn=None,
) -> dict:
    """Phase 4: Map events → fitness scores + alignments.

    Returns dict with keys:
        mapped_events, fitness_raw, fitness_dedup, alignments,
        shadow_count, matched_count, new_deviations
    """
    result = {
        "mapped_events": abstracted_df,
        "fitness_raw": 0.0,
        "fitness_dedup": 0.0,
        "alignments": [],
        "shadow_count": 0,
        "matched_count": 0,
        "new_deviations": set(),
    }

    if abstracted_df.empty or reference_bpmn is None or not agenda_activities:
        return result

    # SBERT mapping
    mapped_events = compliance_engine.map_events_to_agenda(
        abstracted_df, agenda_activities, threshold=sbert_threshold,
    )
    result["mapped_events"] = mapped_events
    if log_fn:
        log_fn(f"Mapped events: {len(mapped_events)}")

    # Count formal vs shadow
    if "mapped_activity" in mapped_events.columns:
        result["shadow_count"] = int(
            mapped_events["mapped_activity"].str.startswith("Deviation:", na=False).sum()
        )
        result["matched_count"] = len(mapped_events) - result["shadow_count"]

    # Convert to event log for PM4Py
    log_data = convert_to_event_log(mapped_events)
    if log_data is None:
        return result

    # Raw fitness (all events)
    compliance_result = compliance_engine.calculate_fitness(reference_bpmn, log_data)
    result["fitness_raw"] = compliance_result.get("score", 0.0)
    result["alignments"] = compliance_result.get("alignments", [])
    if log_fn:
        log_fn(f"Raw fitness: {result['fitness_raw']:.4f}")

    # Dedup fitness (first occurrence per agenda item)
    try:
        act_col = "mapped_activity" if "mapped_activity" in mapped_events.columns else "activity_name"
        formal = mapped_events[
            ~mapped_events[act_col].str.startswith("Deviation:", na=False)
        ].copy()
        if not formal.empty:
            formal["__ts"] = formal["timestamp"].apply(ts_to_seconds)
            formal = formal.sort_values("__ts")
            first = formal.groupby(act_col).first().reset_index()
            first = first.sort_values("__ts").drop(columns=["__ts"])
            first["activity_name"] = first[act_col]
            dedup_log = convert_to_event_log(first)
            if dedup_log is not None and not dedup_log.empty:
                dedup_result = compliance_engine.calculate_fitness(reference_bpmn, dedup_log)
                result["fitness_dedup"] = dedup_result.get("score", 0.0)
                if log_fn:
                    log_fn(f"Dedup fitness: {result['fitness_dedup']:.4f} (trace length: {len(first)})")
    except Exception as e:
        if log_fn:
            log_fn(f"Dedup fitness error: {e}")

    return result


def detect_deviations(alignments: list, accepted: set) -> set:
    """Extract shadow activity labels from alignment results."""
    deviations = set()
    for align in alignments:
        for log_move, model_move in align["alignment"]:
            if (model_move is None or model_move == ">>") and log_move is not None and log_move != ">>":
                if log_move not in accepted:
                    deviations.add(log_move)
    return deviations
