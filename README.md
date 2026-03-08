# Meeting Process Twin

**Multimodal Process Mining for Civic Governance Transparency**

The Meeting Process Twin is a research artifact that transforms city council meeting recordings into structured, process-minable event logs --- revealing the gap between what meetings are *supposed* to do (the published agenda) and what they *actually* do (the enacted process). Built as a master's thesis at TUM, it fuses audio, visual, and NLP signals to perform conformance checking, shadow workflow detection, and parliamentary compliance analysis at scale.

**Core finding:** Formal agendas predict only 31% of actual meeting behavior. The other 69% --- shadow workflows --- are overwhelmingly substantive policy work, not procedural noise.

---

## What This Project Does

Given a meeting video and its published agenda, the pipeline:

1. **Extracts multimodal events** from the recording (audio via Whisper, visual via RTMPose, NLP via regex + spaCy)
2. **Fuses modalities** using temporal proximity rules (e.g., a hand-raise + "all in favor" within 5s = Confirmed Vote)
3. **Abstracts events** into formal activity labels using LLM sliding-window classification (GPT-4o-mini) or keyword rules
4. **Maps activities to the agenda** using Sentence-BERT cosine similarity (threshold *t* = 0.35)
5. **Performs conformance checking** via PM4Py token-replay fitness against a BPMN normative model
6. **Detects shadow workflows** --- activities not matching any agenda item --- and classifies them by deviance type
7. **Evaluates parliamentary compliance** by formalizing Robert's Rules of Order as Declare constraints
8. **Discovers structural patterns** in shadow activities using POWL (Partially Ordered Workflow Language)

The result: a structured governance report that distills hours of meeting video into minutes of actionable intelligence.

---

## Key Results (54 Meetings, 103.5 Hours, 4 U.S. Cities)

| Metric | Value |
|--------|-------|
| Total events extracted | 28,256 |
| Mean deduplicated fitness | 0.311 (agendas predict ~31% of behavior) |
| Shadow activity prevalence | 49.2% of all events |
| Innovation rate (shadow) | 96.7% (substantive policy work) |
| Modality split | Audio 57.0%, Visual 3.8%, Fusion 39.2% |
| Declare compliance | Bimodal: 16 A-grades, 37 F-grades |
| POWL shadow patterns | Isolated 46.1%, Concurrent 29.5%, Sequential 15.1%, Recurring 9.3% |
| Processing speed | 3.5x real-time on CPU (no GPU required) |
| Golden dataset agreement | 75% directional agreement (10 annotated meetings, 9,348 events) |

---

## Architecture

```
Meeting Video ──┬── Whisper ASR (small, 244M) ──┬── NLP Rules (30 patterns) ──┐
                │                                │                             │
                └── RTMPose (ONNX) ──────────────┴── Temporal Fusion ──────────┘
                                                           │
                                                    Raw Event Log
                                                           │
                                    ┌──────────────────────┤
                                    │                      │
                              LLM Abstraction        Keyword Rules
                            (GPT-4o-mini)          (LLM-free fallback)
                                    │                      │
                                    └──────────┬───────────┘
                                               │
                                         SBERT Mapping
                                        (t = 0.35)
                                               │
                                    ┌──────────┴──────────┐
                                    │                     │
                              Formal Events         Shadow Events
                                    │                     │
                            ┌───────┤               ┌─────┤
                            │       │               │     │
                         Token    Declare        Deviance  POWL
                         Replay   Constraints   Taxonomy  Patterns
                         Fitness  (Robert's     (4-cat)   (4-type)
                                   Rules)
                                    │
                              Conformance Report
                                    │
                            Citizen Dashboard
                              (Streamlit)
```

---

## Project Structure

### Core Pipeline

| File | Purpose |
|------|---------|
| `video_processor.py` | Phase 1: Audio extraction, Whisper transcription, RTMPose pose estimation, NLP keyword detection, multimodal fusion |
| `compliance_engine.py` | Phase 2-3: LLM sliding-window abstraction, SBERT mapping, PM4Py fitness, Declare constraints |
| `bpmn_gen.py` | BPMN model generation (agenda -> normative model), Inductive Miner discovery, colored compliance visualization |
| `app.py` | Streamlit citizen-facing dashboard |

### Batch Processing & Analysis

| File | Purpose |
|------|---------|
| `batch_analyze.py` | Full pipeline runner for 54-meeting corpus |
| `generate_figures.py` | Generates 12 thesis figures from conformance.json files |
| `stratified_analysis.py` | Fitness stratified by agenda complexity (sparse/moderate/dense) |
| `sbert_sensitivity.py` | Threshold sensitivity analysis across 10 values (0.15-0.60) |
| `golden_comparison.py` | Golden dataset validation (pipeline vs. human annotations) |
| `rerun_deviance.py` | Re-runs deviance classification without API calls |

### Research Modules

| Directory | Purpose |
|-----------|---------|
| `research/declare/` | Declare constraint formalization of Robert's Rules, conformance checking, violation analysis |
| `research/deviance/` | Four-category deviance taxonomy classifier, deviance reporting |
| `research/powl/` | POWL-based shadow pattern discovery (isolated, concurrent, sequential, recurring) |
| `research/diarization/` | Speaker diarization prototype (WhisperX + pyannote) --- not integrated into main pipeline |
| `research/longitudinal/` | Drift detection and cross-meeting comparison |
| `research/structured_extraction/` | LLM-based structured event extraction with schema validation |

### Data

| Directory | Purpose |
|-----------|---------|
| `meetings/` | 54 meeting directories (3 Alameda, 11 Boston, 15 Denver, 25 Seattle) |
| `meetings/_aggregated/` | Aggregated metrics CSV, corpus-level statistics |
| `Golden-dataset/` | 10 manually annotated Seattle meetings (9,348 segment-level annotations) |
| `thesis/` | LaTeX chapter files and figures for the thesis |

---

## Three Abstraction Strategies

The pipeline supports three event abstraction strategies with different trade-offs:

| Strategy | Method | Dedup Fitness | API Required | Best For |
|----------|--------|---------------|--------------|----------|
| **A** | Keyword rules + SBERT | 68.2% | No | Offline/free deployments |
| **B** | LLM sliding-window (GPT-4o-mini) | 72.7% | Yes | Maximum accuracy |
| **C** | LLM + SBERT (combined) | 31.1% (corpus mean) | Yes | Large-scale batch analysis |

Strategy C (used for the 54-meeting evaluation) combines LLM contextual understanding with SBERT threshold-based formal/shadow partitioning.

---

## Installation

### Prerequisites
- Python 3.10 or 3.11
- FFmpeg (for audio extraction)
- OpenAI API key (for LLM abstraction; optional if using Strategy A)

### Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

No GPU required. The pipeline runs entirely on CPU using ONNX Runtime for pose estimation and Whisper small for transcription.

---

## Usage

### Interactive Dashboard

```bash
streamlit run app.py
```

Upload a meeting video and agenda text, configure parameters in the sidebar, and receive a real-time conformance report with shadow workflow visualization.

### Single Meeting (Headless)

```bash
# Full pipeline (Whisper + LLM abstraction)
python test_pipeline.py --api-key sk-... \
    --video meeting.mp4 --agenda agenda.txt --output-dir ./results

# Skip Whisper (reuse cached transcription)
python test_pipeline.py --api-key sk-... \
    --video meeting.mp4 --agenda agenda.txt --output-dir ./results \
    --skip-video-processing --window-seconds 60 --overlap-ratio 0.5

# LLM-free mode (no API key needed)
python run_no_api.py
```

### Batch Analysis (Full Corpus)

```bash
# Process all meetings
python batch_analyze.py --api-key sk-...

# Generate thesis figures
python generate_figures.py

# Run sensitivity analysis
python sbert_sensitivity.py

# Validate against golden dataset
python golden_comparison.py
```

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `window_seconds` | 60 | LLM classification window size |
| `overlap_ratio` | 0.5 | Window overlap (higher = more context, slower) |
| `min_events_per_window` | 5 | Minimum events to classify a window |
| `sbert_threshold` | 0.35 | Formal/shadow partition threshold |
| `sbert_model` | all-MiniLM-L6-v2 | SBERT model (or all-mpnet-base-v2 for higher accuracy) |
| `min_label_support` | 1 | Minimum occurrences to keep an activity label |

---

## Computational Requirements

Tested on: Windows 11, Intel Core i7, 32 GB RAM, **no GPU**.

| Phase | Time (1h meeting) | Share |
|-------|-------------------|-------|
| Whisper transcription (CPU) | ~12 min | 60% |
| RTMPose visual extraction | ~4 min | 20% |
| LLM abstraction (API) | ~2 min | 10% |
| Audio extraction + NLP + fusion | ~2 min | 10% |
| **Total** | **~20 min** | |

The full 54-meeting corpus (103.5 hours) processed in ~29.5 hours on CPU.

---

## The Thesis

This project is the artifact for a master's thesis at the Technical University of Munich (TUM):

> **Shadow Workflows in Civic Governance: A Multimodal Process Mining Approach to Meeting Analysis**

The thesis demonstrates that:

- **Process mining can be extended to unstructured multimodal domains** by constructing event logs from video recordings
- **Formal agendas predict only 31% of meeting behavior**, quantifying the ostensive-performative gap (Feldman & Pentland, 2003) in civic governance at scale
- **Shadow activities are responsive governance**, not dysfunction --- 96.7% are substantive policy discussions
- **Declare and POWL formalisms** are well-suited to parliamentary analysis, revealing structural patterns in informal meeting behavior
- **Citizens can review meeting outcomes in 4-5 minutes** instead of watching 1.5-2 hours of video per meeting

### Thesis Chapters

1. Introduction
2. Theoretical Foundations (BPM, multimodal fusion, shadow workflows, routine dynamics, Declare, POWL, civic governance, related work)
3. Research Methodology (DSR framework, data collection, golden dataset, evaluation design)
4. Artifact Design (4-phase pipeline, computational requirements)
5. Evaluation (conformance, shadow analysis, sensitivity, golden validation)
6. Discussion (findings, implications, limitations, threats to validity)
7. Conclusion and Future Work

---

## Citation

If you use this work, please cite:

```
Meeting Process Twin: Multimodal Process Mining for Civic Governance Transparency.
Master's Thesis, Technical University of Munich, 2026.
```

---

## License

Research artifact. Contact the author for licensing inquiries.
