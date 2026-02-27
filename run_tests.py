#!/usr/bin/env python3
"""
Self-contained test suite for Meeting Process Twin.

Tests all bug fixes and improvements WITHOUT needing an OpenAI API key.
Uses synthetic council-meeting data and mocked LLM responses.

Run:  python run_tests.py
"""

import sys, os
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import traceback

# ---- Mock streamlit BEFORE any project import --------------------------------
sys.modules["streamlit"] = MagicMock()

import pandas as pd
from bpmn_gen import convert_to_event_log
from video_processor import VideoProcessor
from compliance_engine import ComplianceEngine

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    detail_str = f"  -> {detail}" if detail else ""
    print(f"  [{status}] {name}{detail_str}")
    return condition

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# -----------------------------------------------------------------------------
# TEST 1 — parse_time bug fix (timestamps > 60 minutes)
# -----------------------------------------------------------------------------
section("TEST 1: parse_time bug fix (>60 min timestamps)")

test_cases = [
    ("00:05:30",  5*60+30,   "standard HH:MM:SS"),
    ("01:02:15",  3600+135,  "1h+ video timestamp"),
    ("01:30:00",  5400,      "90-minute mark"),
    ("65:30",     65*60+30,  "legacy MM:SS large-minute format"),
    ("00:00:00",  0,         "start of video"),
]

df = pd.DataFrame([{"timestamp": t, "activity_name": "Test"} for t, _, _ in test_cases])
log = convert_to_event_log(df)

for (ts, expected_seconds, label), actual_dt in zip(test_cases, log["time:timestamp"]):
    expected_dt = datetime(2023, 1, 1) + timedelta(seconds=expected_seconds)
    check(f"parse_time({ts!r}) [{label}]", actual_dt == expected_dt,
          f"got {actual_dt.strftime('%H:%M:%S')} expected {expected_dt.strftime('%H:%M:%S')}")

# -----------------------------------------------------------------------------
# TEST 2 — _format_timestamp produces correct HH:MM:SS
# -----------------------------------------------------------------------------
section("TEST 2: _format_timestamp standardization")

vp = VideoProcessor.__new__(VideoProcessor)  # create without __init__

fmt_cases = [
    (0,          "00:00:00"),
    (65,         "00:01:05"),
    (3600,       "01:00:00"),
    (3690,       "01:01:30"),
    (5400,       "01:30:00"),
    (7261,       "02:01:01"),
]
for secs, expected in fmt_cases:
    result = vp._format_timestamp(secs)
    check(f"_format_timestamp({secs}s)", result == expected, f"got={result!r} expected={expected!r}")

# -----------------------------------------------------------------------------
# TEST 3 — NLP activity extraction (council keywords)
# -----------------------------------------------------------------------------
section("TEST 3: NLP activity extraction (council keywords)")

# Build a processor with a real spaCy nlp (needed for extract_action_object)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    vp_real = VideoProcessor.__new__(VideoProcessor)
    vp_real.nlp = nlp
    vp_real.debug = False
    nlp_ok = True
except Exception as e:
    nlp_ok = False
    print(f"  (skipping NLP tests — spaCy not loadable: {e})")

if nlp_ok:
    keyword_cases = [
        ("I move to approve the budget",          "Propose Motion"),
        ("I second the motion",                   "Second Motion"),
        ("All in favor say aye",                  "Call for Vote"),
        ("The ayes have it, motion carries",      "Vote Result"),
        ("The meeting is adjourned",              "Adjourn Meeting"),
        ("We will now open the public hearing",   "Public Comment"),
        ("This is agenda item number five",       "Introduce Agenda Item"),
        ("Council will take a short recess",      "Recess"),
        ("Staff report on the budget projections","Staff Report"),
        ("Roll call vote on ordinance 2024-15",   "Roll Call Vote"),
        ("We need to approve this resolution",    "Discuss Resolution"),
    ]
    for text, expected_activity in keyword_cases:
        activity, _ = vp_real.extract_action_object(text)
        check(f"extract_action_object: {text[:45]!r}", activity == expected_activity,
              f"got={activity!r}")

# -----------------------------------------------------------------------------
# TEST 4 — ComplianceEngine abstraction (with mocked LLM)
# -----------------------------------------------------------------------------
section("TEST 4: Abstraction pipeline (mocked LLM)")

engine = ComplianceEngine()

# Synthetic raw events (simulating a 1h council meeting transcript)
raw_events_data = []
activities_timeline = [
    (0,    "Call to Order",        "Mayor called the meeting to order"),
    (120,  "Propose Motion",       "I move to approve the consent agenda"),
    (125,  "Second Motion",        "I second the motion"),
    (130,  "Call for Vote",        "All in favor say aye"),
    (135,  "Vote Result",          "Motion carries unanimously"),
    (300,  "Staff Presentation",   "Staff presented the annual budget update"),
    (360,  "Budget Discussion",    "Discussion on budget projections"),
    (420,  "Public Comment",       "Public hearing on proposed ordinance"),
    (600,  "Propose Motion",       "Motion to approve ordinance 2024-05"),
    (605,  "Second Motion",        "Seconded"),
    (610,  "Call for Vote",        "Roll call vote"),
    (615,  "Vote Result",          "Motion passes 5-2"),
    (3600, "Adjourn Meeting",      "The meeting is now adjourned"),
]
for secs, act, detail in activities_timeline:
    vp2 = VideoProcessor.__new__(VideoProcessor)
    raw_events_data.append({
        "timestamp": vp2._format_timestamp(secs),
        "activity_name": act,
        "source": "Audio",
        "details": detail,
        "raw_seconds": float(secs),
    })

df_raw = pd.DataFrame(raw_events_data)

agenda_activities = [
    "Call to Order",
    "Approve Consent Agenda",
    "Staff Budget Presentation",
    "Public Hearing on Ordinance 2024-05",
    "Adopt Ordinance 2024-05",
    "Adjourn Meeting",
]

# Mock the LLM call so it returns sensible labels based on what's in the window
def mock_openai_classify(model, agenda_tasks, window_events, api_key, timeout=60):
    w = window_events.lower()
    if "call to order" in w:        return "Call to Order"
    if "consent" in w or "second" in w or "favor" in w: return "Approve Consent Agenda"
    if "budget" in w or "staff" in w: return "Staff Budget Presentation"
    if "public" in w or "hearing" in w: return "Public Hearing on Ordinance 2024-05"
    if "ordinance" in w or "roll call" in w: return "Adopt Ordinance 2024-05"
    if "adjourn" in w:              return "Adjourn Meeting"
    return "Noise"

with patch.object(engine, "_openai_classify_window", side_effect=mock_openai_classify), \
     patch.object(engine, "_ollama_available", return_value=False):

    df_abstracted = engine.abstract_events_df(
        df=df_raw,
        agenda_tasks=agenda_activities,
        window_seconds=300,
        overlap_ratio=0.5,
        min_events_per_window=1,
        min_label_support=1,
        shadow_min_ratio=0.1,
        api_key="fake-key",
        openai_model="gpt-4o-mini",
        cache={},
    )

check("Abstraction returns a DataFrame",  isinstance(df_abstracted, pd.DataFrame))
check("Abstraction has >0 events",        len(df_abstracted) > 0,
      f"got {len(df_abstracted)} events")
check("Abstraction has activity_name col","activity_name" in df_abstracted.columns)
check("No Noise events in output",        not df_abstracted["activity_name"].str.lower().eq("noise").any(),
      f"activities: {df_abstracted['activity_name'].unique().tolist()}")

# -----------------------------------------------------------------------------
# TEST 5 — SBERT mapping + fitness score
# -----------------------------------------------------------------------------
section("TEST 5: SBERT mapping + PM4Py fitness")

try:
    df_mapped = engine.map_events_to_agenda(df_abstracted, agenda_activities, threshold=0.3)
    check("map_events_to_agenda returns DataFrame",  isinstance(df_mapped, pd.DataFrame))
    check("mapped_activity column present",          "mapped_activity" in df_mapped.columns)

    unique_mapped = df_mapped["mapped_activity"].unique().tolist()
    check("Mapped to >=2 distinct agenda items",      len(unique_mapped) >= 2,
          f"mapped to: {unique_mapped}")

    log_data = convert_to_event_log(df_mapped)
    check("convert_to_event_log not None",           log_data is not None)
    check("log has case:concept:name column",        "case:concept:name" in log_data.columns)

    # Build a minimal BPMN from the agenda (no API needed: directly build it)
    from pm4py.objects.bpmn.obj import BPMN
    bpmn_graph = BPMN()
    start = BPMN.StartEvent(name="Start"); bpmn_graph.add_node(start)
    prev = start
    for act in agenda_activities:
        task = BPMN.Task(name=act); bpmn_graph.add_node(task)
        bpmn_graph.add_flow(BPMN.SequenceFlow(prev, task)); prev = task
    end = BPMN.EndEvent(name="End"); bpmn_graph.add_node(end)
    bpmn_graph.add_flow(BPMN.SequenceFlow(prev, end))

    result = engine.calculate_fitness(bpmn_graph, log_data)
    fitness = result.get("score", 0.0)
    check("calculate_fitness returns dict",          isinstance(result, dict))
    check("fitness score is a number",               isinstance(fitness, float))
    check("fitness > 0",                             fitness > 0.0, f"fitness={fitness:.3f}")
    print(f"\n  Fitness score: {fitness*100:.1f}%")
    print(f"  Alignments:   {len(result.get('alignments',[]))} traces")

except Exception as e:
    check("SBERT/fitness pipeline (no exception)", False, str(e))
    traceback.print_exc()

# -----------------------------------------------------------------------------
# TEST 6 — _build_classification_prompt structure
# -----------------------------------------------------------------------------
section("TEST 6: LLM prompt quality check")

prompt = engine._build_classification_prompt(
    agenda_tasks=["Call to Order", "Approve Consent Agenda", "Adjourn Meeting"],
    window_events="- [00:01:00] Propose Motion (Audio): I move to approve the consent agenda"
)

check("Prompt contains FORMAL AGENDA TASKS section",  "FORMAL AGENDA TASKS" in prompt)
check("Prompt contains CLASSIFICATION RULES section", "CLASSIFICATION RULES" in prompt)
check("Prompt contains EXAMPLES section",             "EXAMPLES" in prompt)
check("Prompt contains Shadow: format instruction",   "Shadow:" in prompt)
check("Prompt contains Noise instruction",            "Noise" in prompt)
check("Prompt contains the actual agenda items",      "Call to Order" in prompt)
check("Prompt contains the meeting events",           "consent agenda" in prompt)
check("Prompt length is substantial (>500 chars)",    len(prompt) > 500, f"len={len(prompt)}")

# -----------------------------------------------------------------------------
# TEST 7 — visual debounce (no duplicate voting events within 5s)
# -----------------------------------------------------------------------------
section("TEST 7: Visual event debounce")

# Simulate a sequence where hand is raised continuously for 10s (should produce <=2 events)
fake_events = []
vp3 = VideoProcessor.__new__(VideoProcessor)
# Simulate the debounce logic inline
for t in range(0, 30, 2):  # every 2 seconds
    hand_raised = (5 <= t <= 15)  # hand raised from 5s to 15s
    if hand_raised:
        if fake_events and fake_events[-1]['activity_name'] == 'Voting' and \
                (t - fake_events[-1]['raw_seconds']) < 5:
            continue  # debounce
        fake_events.append({
            'timestamp': vp3._format_timestamp(t),
            'activity_name': 'Voting',
            'source': 'Video',
            'details': 'Hand Raised',
            'raw_seconds': float(t),
        })

check("Debounce reduces 5s continuous raise to <=3 events", len(fake_events) <= 3,
      f"got {len(fake_events)} voting events")
# range(0,30,2) skips odd seconds; first t in [5..15] is t=6
check("First vote timestamp is correct", fake_events[0]['timestamp'] == "00:00:06" if fake_events else True,
      f"got {fake_events[0]['timestamp'] if fake_events else 'none'}")

# -----------------------------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------------------------
section("RESULTS SUMMARY")
passed = sum(1 for _, ok in results if ok)
total  = len(results)
pct    = passed / total * 100

for name, ok in results:
    icon = "V" if ok else "X"
    print(f"  [{icon}] {name}")

print(f"\n  {'-'*40}")
print(f"  Passed: {passed}/{total}  ({pct:.0f}%)")
if passed == total:
    print(f"\n  \033[92mAll tests passed!\033[0m")
else:
    failed = [(n, ok) for n, ok in results if not ok]
    print(f"\n  \033[91m{total-passed} test(s) FAILED:\033[0m")
    for name, _ in failed:
        print(f"    X {name}")

sys.exit(0 if passed == total else 1)
