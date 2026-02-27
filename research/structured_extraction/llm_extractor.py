"""GPT-4o-mini structured event extraction from meeting transcripts.

Replaces/supplements keyword rules with richer (actor, action, object) tuples.
Uses OpenAI structured output (JSON mode) for reliable parsing.

Example usage:
    from research.structured_extraction.llm_extractor import StructuredEventExtractor

    extractor = StructuredEventExtractor(api_key="sk-...")
    results = extractor.extract_events(
        transcript_segments=[
            {"start": 0, "end": 30, "text": "I call this meeting to order.", "speaker": "SPEAKER_00"},
            {"start": 30, "end": 60, "text": "Please call the roll.", "speaker": "SPEAKER_00"},
        ],
        agenda_items=["Call to Order", "Roll Call", "Public Comment"],
    )
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project-level time utilities
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.time_utils import ts_to_seconds, seconds_to_ts  # noqa: E402

# ---------------------------------------------------------------------------
# OpenAI import
# ---------------------------------------------------------------------------
try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Schema imports
# ---------------------------------------------------------------------------
from research.structured_extraction.schema import (  # noqa: E402
    MeetingEvent,
    ExtractionResult,
    ExtractionBatch,
)

logger = logging.getLogger(__name__)


class StructuredEventExtractor:
    """LLM-based structured event extraction from meeting transcripts.

    Uses GPT-4o-mini (or another OpenAI model) with JSON mode to extract
    structured (actor, action, object) event tuples from windowed meeting
    transcript segments.

    Attributes:
        client: OpenAI client instance.
        model:  Model name for extraction (default: gpt-4o-mini).
    """

    # Default system prompt for structured extraction
    SYSTEM_PROMPT = (
        "You are an expert analyst of government council meeting transcripts. "
        "Your task is to extract structured events from transcript windows.\n\n"
        "For each distinct event in the transcript, extract:\n"
        "- actor: The person or role performing the action (Mayor, Council Member X, City Clerk, etc.)\n"
        "- action: The verb phrase (called to order, moved to approve, opened hearing, etc.)\n"
        "- object: The target of the action (consent agenda, Resolution 2026-042, etc.)\n"
        "- event_type: One of: formal, shadow, procedural, noise\n"
        "- confidence: Your confidence (0.0-1.0) in this extraction\n"
        "- timestamp: When this event occurred (from the transcript timestamps)\n"
        "- raw_text: The relevant transcript excerpt\n\n"
        "CLASSIFICATION RULES:\n"
        "- formal: Event directly relates to a planned agenda item\n"
        "- procedural: Parliamentary procedure (seconds, votes, points of order) "
        "not tied to a specific agenda item\n"
        "- shadow: Off-agenda activity (sidebar conversations, informal agreements, "
        "unscheduled breaks)\n"
        "- noise: Transcription artifacts, filler words, irrelevant chatter\n\n"
        "IMPORTANT:\n"
        "- Extract EVERY meaningful event. Do not skip procedural events.\n"
        "- Use exact role titles when identifiable (Mayor, Council Member [Name], etc.)\n"
        "- If the speaker is unknown, set actor to null.\n"
        "- Set confidence lower for ambiguous or unclear events.\n"
        "- Return a JSON object with a single key 'events' containing an array of event objects.\n"
    )

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.1,
    ) -> None:
        """Initialize the structured event extractor.

        Args:
            api_key:     OpenAI API key.
            model:       Model name (default: gpt-4o-mini).
            max_retries: Number of retries on API failure.
            retry_delay: Base delay in seconds between retries (exponential backoff).
            temperature: Sampling temperature for the LLM (lower = more deterministic).

        Raises:
            ImportError: If the openai library is not installed.
        """
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai library is not installed. Install with: pip install openai"
            )

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self._total_tokens_used = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_events(
        self,
        transcript_segments: list[dict],
        agenda_items: list[str],
        window_seconds: int = 120,
        overlap_seconds: int = 30,
        min_segments_per_window: int = 1,
    ) -> ExtractionBatch:
        """Extract structured events from transcript using LLM.

        The transcript is divided into overlapping time windows. Each window
        is sent to the LLM for extraction. Results are deduplicated across
        overlapping windows.

        Args:
            transcript_segments: List of dicts with keys: start (float seconds),
                                 end (float seconds), text (str), and optionally
                                 speaker (str).
            agenda_items:        List of formal agenda item labels.
            window_seconds:      Duration of each extraction window in seconds.
            overlap_seconds:     Overlap between consecutive windows in seconds.
            min_segments_per_window: Minimum transcript segments required per
                                     window to trigger extraction.

        Returns:
            ExtractionBatch containing all extracted events across all windows.
        """
        if not transcript_segments:
            logger.warning("No transcript segments provided.")
            return ExtractionBatch(agenda_items=agenda_items)

        # Determine time range
        all_starts = [seg.get("start", 0) for seg in transcript_segments]
        all_ends = [seg.get("end", 0) for seg in transcript_segments]
        total_start = min(all_starts)
        total_end = max(all_ends)

        logger.info(
            "Extracting events from %.1fs to %.1fs (%d segments, window=%ds, overlap=%ds)",
            total_start, total_end, len(transcript_segments),
            window_seconds, overlap_seconds,
        )

        # Build windows
        windows = self._build_windows(
            transcript_segments,
            total_start,
            total_end,
            window_seconds,
            overlap_seconds,
            min_segments_per_window,
        )

        logger.info("Created %d extraction windows.", len(windows))

        # Process each window
        results: list[ExtractionResult] = []
        for i, (w_start, w_end, w_segments) in enumerate(windows):
            logger.info(
                "Processing window %d/%d [%s - %s] (%d segments)",
                i + 1, len(windows),
                seconds_to_ts(int(w_start)), seconds_to_ts(int(w_end)),
                len(w_segments),
            )

            result = self._extract_window(w_start, w_end, w_segments, agenda_items)
            if result is not None:
                results.append(result)

        batch = ExtractionBatch(
            results=results,
            agenda_items=agenda_items,
        )

        # Deduplicate across overlapping windows
        batch = self._deduplicate_batch(batch)

        logger.info(
            "Extraction complete: %d events across %d windows (tokens: %d)",
            batch.total_events, len(batch.results), self._total_tokens_used,
        )

        return batch

    def extract_single_window(
        self,
        text: str,
        window_start: str,
        window_end: str,
        agenda_items: list[str],
    ) -> ExtractionResult:
        """Extract events from a single text window (convenience method).

        Args:
            text:         Raw transcript text for this window.
            window_start: Window start timestamp (HH:MM:SS).
            window_end:   Window end timestamp (HH:MM:SS).
            agenda_items: List of formal agenda item labels.

        Returns:
            ExtractionResult for this window.
        """
        prompt = self._build_extraction_prompt(text, agenda_items)
        response = self._call_llm(prompt)
        events = self._parse_response(response, window_start)

        return ExtractionResult(
            events=events,
            window_start=window_start,
            window_end=window_end,
            model=self.model,
            token_usage=response.get("_token_usage", 0),
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_extraction_prompt(
        self,
        window_text: str,
        agenda_items: list[str],
    ) -> str:
        """Build few-shot extraction prompt for a transcript window.

        Args:
            window_text:  The transcript text for the current window.
            agenda_items: List of formal agenda item labels.

        Returns:
            The user-message prompt string to send to the LLM.
        """
        agenda_str = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(agenda_items))

        few_shot_example = json.dumps({
            "events": [
                {
                    "timestamp": "00:01:15",
                    "actor": "Mayor Johnson",
                    "action": "called the meeting to order",
                    "object": None,
                    "event_type": "formal",
                    "confidence": 0.95,
                    "raw_text": "Mayor Johnson: I'd like to call this meeting to order at 7:01 PM."
                },
                {
                    "timestamp": "00:01:30",
                    "actor": "City Clerk",
                    "action": "called the roll",
                    "object": "council members",
                    "event_type": "formal",
                    "confidence": 0.90,
                    "raw_text": "City Clerk: I'll now call the roll. Council Member Adams?"
                },
                {
                    "timestamp": "00:01:45",
                    "actor": "Council Member Adams",
                    "action": "responded present",
                    "object": None,
                    "event_type": "procedural",
                    "confidence": 0.85,
                    "raw_text": "Here."
                },
            ]
        }, indent=2)

        prompt = (
            f"FORMAL AGENDA ITEMS for this meeting:\n"
            f"{agenda_str}\n\n"
            f"FEW-SHOT EXAMPLE (for format reference only):\n"
            f"```json\n{few_shot_example}\n```\n\n"
            f"Now extract ALL meaningful events from the following transcript window. "
            f"Return ONLY a JSON object with key 'events'.\n\n"
            f"TRANSCRIPT WINDOW:\n"
            f"---\n"
            f"{window_text}\n"
            f"---\n"
        )

        return prompt

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> dict:
        """Call the OpenAI API with retry logic.

        Args:
            user_prompt: The user message to send.

        Returns:
            Parsed JSON response dict with an additional '_token_usage' key.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    max_tokens=4096,
                )

                content = response.choices[0].message.content
                token_usage = 0
                if response.usage:
                    token_usage = response.usage.total_tokens
                    self._total_tokens_used += token_usage

                parsed = json.loads(content)
                parsed["_token_usage"] = token_usage
                return parsed

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    "JSON parse error on attempt %d/%d: %s",
                    attempt + 1, self.max_retries, e,
                )
            except openai.RateLimitError as e:
                last_error = e
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "Rate limit hit on attempt %d/%d. Waiting %.1fs...",
                    attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)
            except openai.APIError as e:
                last_error = e
                logger.warning(
                    "API error on attempt %d/%d: %s",
                    attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        raise RuntimeError(
            f"LLM extraction failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: dict,
        fallback_timestamp: str = "00:00:00",
    ) -> list[MeetingEvent]:
        """Parse LLM JSON response into a list of MeetingEvent objects.

        Handles malformed events gracefully by logging warnings and skipping
        invalid entries rather than failing the entire window.

        Args:
            response:           Parsed JSON dict from the LLM.
            fallback_timestamp: Default timestamp for events missing one.

        Returns:
            List of validated MeetingEvent objects.
        """
        raw_events = response.get("events", [])
        if not isinstance(raw_events, list):
            logger.warning("LLM response 'events' is not a list: %s", type(raw_events))
            return []

        parsed_events: list[MeetingEvent] = []

        for i, raw in enumerate(raw_events):
            if not isinstance(raw, dict):
                logger.warning("Skipping non-dict event at index %d: %s", i, type(raw))
                continue

            try:
                # Normalize fields
                event = MeetingEvent(
                    timestamp=raw.get("timestamp", fallback_timestamp),
                    actor=raw.get("actor"),
                    action=raw.get("action", "unknown action"),
                    object=raw.get("object"),
                    event_type=self._normalize_event_type(raw.get("event_type", "noise")),
                    confidence=self._clamp_confidence(raw.get("confidence", 0.5)),
                    raw_text=raw.get("raw_text", ""),
                )
                parsed_events.append(event)

            except Exception as e:
                logger.warning(
                    "Failed to parse event at index %d: %s. Raw: %s",
                    i, e, raw,
                )

        return parsed_events

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def _build_windows(
        self,
        segments: list[dict],
        total_start: float,
        total_end: float,
        window_seconds: int,
        overlap_seconds: int,
        min_segments_per_window: int,
    ) -> list[tuple[float, float, list[dict]]]:
        """Divide transcript segments into overlapping time windows.

        Args:
            segments:              All transcript segments.
            total_start:           Start time of the transcript.
            total_end:             End time of the transcript.
            window_seconds:        Window duration in seconds.
            overlap_seconds:       Overlap between consecutive windows.
            min_segments_per_window: Minimum segments to include a window.

        Returns:
            List of (start_sec, end_sec, window_segments) tuples.
        """
        step = max(1, window_seconds - overlap_seconds)
        windows: list[tuple[float, float, list[dict]]] = []

        current = total_start
        while current < total_end:
            w_end = min(current + window_seconds, total_end)

            # Collect segments that overlap with this window
            w_segments = [
                seg for seg in segments
                if seg.get("end", 0) > current and seg.get("start", 0) < w_end
            ]

            if len(w_segments) >= min_segments_per_window:
                windows.append((current, w_end, w_segments))

            current += step

        return windows

    def _extract_window(
        self,
        w_start: float,
        w_end: float,
        segments: list[dict],
        agenda_items: list[str],
    ) -> Optional[ExtractionResult]:
        """Extract events from a single time window.

        Args:
            w_start:      Window start in seconds.
            w_end:        Window end in seconds.
            segments:     Transcript segments within this window.
            agenda_items: Agenda item labels.

        Returns:
            ExtractionResult, or None if extraction fails.
        """
        # Build window text with speaker labels and timestamps
        text_lines = []
        for seg in segments:
            ts = seconds_to_ts(int(seg.get("start", 0)))
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            if text:
                text_lines.append(f"[{ts}] {speaker}: {text}")

        window_text = "\n".join(text_lines)

        if not window_text.strip():
            return None

        prompt = self._build_extraction_prompt(window_text, agenda_items)

        try:
            response = self._call_llm(prompt)
        except RuntimeError as e:
            logger.error("LLM extraction failed for window [%s-%s]: %s",
                         seconds_to_ts(int(w_start)), seconds_to_ts(int(w_end)), e)
            return None

        events = self._parse_response(response, fallback_timestamp=seconds_to_ts(int(w_start)))
        token_usage = response.get("_token_usage", 0)

        return ExtractionResult(
            events=events,
            window_start=seconds_to_ts(int(w_start)),
            window_end=seconds_to_ts(int(w_end)),
            model=self.model,
            token_usage=token_usage,
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def _deduplicate_batch(self, batch: ExtractionBatch) -> ExtractionBatch:
        """Remove duplicate events caused by window overlap.

        Events are considered duplicates if they share the same timestamp,
        action, and actor (or very similar raw_text within a 5-second window).
        When duplicates are found, the one with higher confidence is kept.

        Args:
            batch: ExtractionBatch with potential duplicates.

        Returns:
            Deduplicated ExtractionBatch.
        """
        all_events = batch.all_events

        if not all_events:
            return batch

        # Sort by timestamp, then by confidence (descending)
        all_events.sort(key=lambda e: (e.timestamp, -e.confidence))

        seen: set[str] = set()
        deduped: list[MeetingEvent] = []

        for event in all_events:
            # Create a dedup key from actor + action + timestamp (within 5s bucket)
            ts_sec = ts_to_seconds(event.timestamp)
            ts_bucket = (ts_sec // 5) * 5  # 5-second buckets
            key = f"{event.actor or 'none'}|{event.action.lower().strip()}|{ts_bucket}"

            if key not in seen:
                seen.add(key)
                deduped.append(event)

        # Reconstruct into a single ExtractionResult
        if deduped:
            result = ExtractionResult(
                events=deduped,
                window_start=batch.results[0].window_start if batch.results else "00:00:00",
                window_end=batch.results[-1].window_end if batch.results else "00:00:00",
                model=self.model,
                token_usage=batch.total_tokens,
            )
            return ExtractionBatch(
                results=[result],
                meeting_id=batch.meeting_id,
                agenda_items=batch.agenda_items,
            )

        return batch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_event_type(raw_type: str) -> str:
        """Normalize event type string to one of the valid literals."""
        mapping = {
            "formal": "formal",
            "shadow": "shadow",
            "procedural": "procedural",
            "noise": "noise",
            # Common LLM variations
            "agenda": "formal",
            "off-agenda": "shadow",
            "informal": "shadow",
            "procedure": "procedural",
            "parliamentary": "procedural",
            "artifact": "noise",
            "filler": "noise",
        }
        return mapping.get(raw_type.lower().strip(), "noise")

    @staticmethod
    def _clamp_confidence(value) -> float:
        """Clamp confidence to [0.0, 1.0] range."""
        try:
            v = float(value)
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return 0.5

    @property
    def total_tokens_used(self) -> int:
        """Total tokens consumed across all extraction calls in this session."""
        return self._total_tokens_used

    def __repr__(self) -> str:
        return f"StructuredEventExtractor(model={self.model!r})"
