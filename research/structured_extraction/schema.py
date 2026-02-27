"""Pydantic models for structured meeting event extraction.

Defines the schema for LLM-extracted meeting events, providing type safety,
validation, and serialization for the structured extraction pipeline.

Example usage:
    from research.structured_extraction.schema import MeetingEvent, ExtractionResult

    event = MeetingEvent(
        timestamp="00:15:30",
        actor="Council Member Smith",
        action="moved to approve",
        object="consent agenda items A through D",
        event_type="formal",
        confidence=0.92,
        raw_text="Council Member Smith moved to approve consent agenda items A through D.",
    )

    result = ExtractionResult(
        events=[event],
        window_start="00:15:00",
        window_end="00:17:00",
    )
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class MeetingEvent(BaseModel):
    """A single structured meeting event extracted by LLM.

    Represents an (actor, action, object) tuple with metadata for
    event classification and provenance tracking.

    Attributes:
        timestamp:   Event timestamp in HH:MM:SS format.
        actor:       The person or role performing the action (e.g. "Mayor",
                     "Council Member Garcia", "City Clerk"). None if unknown.
        action:      The verb phrase describing what happened (e.g.
                     "called the meeting to order", "moved to approve").
        object:      The target or subject of the action (e.g.
                     "Resolution 2026-042", "public comment period"). None if
                     intransitive.
        event_type:  Classification of the event into one of four categories:
                     - ``formal``: Matches a planned agenda item.
                     - ``shadow``: Off-agenda informal workflow.
                     - ``procedural``: Parliamentary procedure (seconds, votes).
                     - ``noise``: Irrelevant chatter, transcription artifacts.
        confidence:  LLM's self-assessed confidence in the extraction (0.0-1.0).
        raw_text:    The original transcript text from which this event was extracted.
        agenda_item: The matched agenda item label (populated post-extraction
                     by SBERT mapping). None until mapping is performed.
    """

    timestamp: str = Field(
        ...,
        description="Event timestamp in HH:MM:SS format",
        examples=["00:05:30", "01:15:00"],
    )
    actor: Optional[str] = Field(
        default=None,
        description="Person or role performing the action",
        examples=["Mayor", "Council Member Smith", "City Clerk"],
    )
    action: str = Field(
        ...,
        description="Verb phrase describing the action",
        examples=["called the meeting to order", "moved to approve", "opened public hearing"],
    )
    object: Optional[str] = Field(
        default=None,
        description="Target or subject of the action",
        examples=["consent agenda", "Resolution 2026-042", "budget amendment"],
    )
    event_type: Literal["formal", "shadow", "procedural", "noise"] = Field(
        ...,
        description="Classification: formal (agenda), shadow (off-agenda), procedural, or noise",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM confidence in this extraction (0.0 to 1.0)",
    )
    raw_text: str = Field(
        ...,
        description="Original transcript text this event was extracted from",
    )
    agenda_item: Optional[str] = Field(
        default=None,
        description="Matched agenda item label (populated by SBERT mapping post-extraction)",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        """Ensure timestamp is in HH:MM:SS or MM:SS format."""
        parts = v.strip().split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Timestamp must be HH:MM:SS or MM:SS, got: {v!r}"
            )
        try:
            nums = [int(p) for p in parts]
        except ValueError:
            raise ValueError(f"Timestamp parts must be integers, got: {v!r}")

        if len(nums) == 3:
            h, m, s = nums
        else:
            h, m, s = 0, nums[0], nums[1]

        if not (0 <= m < 60 and 0 <= s < 60):
            raise ValueError(
                f"Invalid minutes/seconds in timestamp: {v!r}"
            )
        return f"{h:02d}:{m:02d}:{s:02d}"

    @field_validator("action")
    @classmethod
    def validate_action_not_empty(cls, v: str) -> str:
        """Ensure action is not empty or whitespace-only."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Action must not be empty")
        return stripped

    def to_event_row(self) -> dict:
        """Convert to a dict compatible with the standard pipeline DataFrame schema.

        Returns:
            Dict with keys: timestamp, activity_name, source, details,
            original_text, mapped_activity, event_type, confidence, actor.
        """
        # Build a descriptive activity name from (actor, action, object)
        parts = []
        if self.actor:
            parts.append(self.actor)
        parts.append(self.action)
        if self.object:
            parts.append(self.object)
        activity_name = " ".join(parts)

        return {
            "timestamp": self.timestamp,
            "activity_name": activity_name,
            "source": "llm_structured_extraction",
            "details": f"actor={self.actor}, object={self.object}, type={self.event_type}",
            "original_text": self.raw_text,
            "mapped_activity": self.agenda_item or "",
            "event_type": self.event_type,
            "confidence": self.confidence,
            "actor": self.actor or "",
        }


class ExtractionResult(BaseModel):
    """Result of structured event extraction for a single transcript window.

    Groups extracted events with their source window boundaries for
    provenance tracking and windowed processing.

    Attributes:
        events:       List of extracted MeetingEvent objects.
        window_start: Start timestamp of the transcript window (HH:MM:SS).
        window_end:   End timestamp of the transcript window (HH:MM:SS).
        model:        LLM model used for extraction.
        token_usage:  Approximate token count for this extraction call.
    """

    events: list[MeetingEvent] = Field(
        default_factory=list,
        description="Extracted meeting events from this window",
    )
    window_start: str = Field(
        ...,
        description="Start timestamp of the source transcript window (HH:MM:SS)",
    )
    window_end: str = Field(
        ...,
        description="End timestamp of the source transcript window (HH:MM:SS)",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="LLM model used for extraction",
    )
    token_usage: Optional[int] = Field(
        default=None,
        description="Approximate token count consumed by this extraction call",
    )

    @property
    def event_count(self) -> int:
        """Number of events extracted in this window."""
        return len(self.events)

    @property
    def formal_count(self) -> int:
        """Number of formal (agenda-matching) events."""
        return sum(1 for e in self.events if e.event_type == "formal")

    @property
    def shadow_count(self) -> int:
        """Number of shadow (off-agenda) events."""
        return sum(1 for e in self.events if e.event_type == "shadow")

    @property
    def avg_confidence(self) -> float:
        """Average extraction confidence across events."""
        if not self.events:
            return 0.0
        return sum(e.confidence for e in self.events) / len(self.events)

    def to_dataframe_rows(self) -> list[dict]:
        """Convert all events to pipeline-compatible dict rows."""
        return [event.to_event_row() for event in self.events]

    def filter_by_confidence(self, min_confidence: float = 0.5) -> list[MeetingEvent]:
        """Return only events meeting the minimum confidence threshold."""
        return [e for e in self.events if e.confidence >= min_confidence]

    def filter_by_type(self, event_type: str) -> list[MeetingEvent]:
        """Return only events of the specified type."""
        return [e for e in self.events if e.event_type == event_type]


class ExtractionBatch(BaseModel):
    """Collection of ExtractionResults across all windows of a meeting.

    Provides aggregate statistics and DataFrame conversion for the
    full meeting extraction.

    Attributes:
        results:      List of per-window ExtractionResult objects.
        meeting_id:   Identifier for the source meeting.
        agenda_items: The agenda items used during extraction.
    """

    results: list[ExtractionResult] = Field(
        default_factory=list,
        description="Per-window extraction results",
    )
    meeting_id: str = Field(
        default="unknown",
        description="Identifier for the source meeting",
    )
    agenda_items: list[str] = Field(
        default_factory=list,
        description="Agenda items used during extraction",
    )

    @property
    def total_events(self) -> int:
        """Total events extracted across all windows."""
        return sum(r.event_count for r in self.results)

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all extraction calls."""
        return sum(r.token_usage or 0 for r in self.results)

    @property
    def all_events(self) -> list[MeetingEvent]:
        """Flat list of all extracted events across all windows."""
        events = []
        for r in self.results:
            events.extend(r.events)
        return events

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert all extracted events to a single pandas DataFrame."""
        import pandas as pd

        rows = []
        for result in self.results:
            rows.extend(result.to_dataframe_rows())

        if not rows:
            return pd.DataFrame(columns=[
                "timestamp", "activity_name", "source", "details",
                "original_text", "mapped_activity", "event_type",
                "confidence", "actor",
            ])

        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Generate aggregate summary statistics."""
        all_events = self.all_events
        type_counts = {}
        for e in all_events:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        unique_actors = {e.actor for e in all_events if e.actor}

        return {
            "meeting_id": self.meeting_id,
            "total_windows": len(self.results),
            "total_events": self.total_events,
            "total_tokens": self.total_tokens,
            "event_type_distribution": type_counts,
            "unique_actors": sorted(unique_actors),
            "unique_actor_count": len(unique_actors),
            "avg_confidence": (
                sum(e.confidence for e in all_events) / len(all_events)
                if all_events else 0.0
            ),
            "agenda_items_used": len(self.agenda_items),
        }
