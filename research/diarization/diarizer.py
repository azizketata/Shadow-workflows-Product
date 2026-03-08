"""Speaker diarization engine for meeting audio.

Uses WhisperX for word-level aligned transcription and pyannote
for speaker segmentation. Produces speaker-attributed event DataFrames.

Dependencies:
    pip install whisperx
    Requires HuggingFace token for pyannote gated models.
    Accept the pyannote terms at:
        https://huggingface.co/pyannote/speaker-diarization-3.1
        https://huggingface.co/pyannote/segmentation-3.0

Example usage:
    from research.diarization.diarizer import MeetingDiarizer

    diarizer = MeetingDiarizer(hf_token="hf_...", device="cuda")
    segments = diarizer.diarize("meeting_audio.wav")
    df = diarizer.segments_to_dataframe(segments)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import pandas as pd

# --- Lazy / guarded imports for WhisperX -----------------------------------
_WHISPERX_AVAILABLE = False
_WHISPERX_IMPORT_ERROR: Optional[str] = None

try:
    import whisperx
    _WHISPERX_AVAILABLE = True
except ImportError as exc:
    _WHISPERX_IMPORT_ERROR = (
        f"whisperx is not installed. Install it with:\n"
        f"  pip install whisperx\n"
        f"You also need a HuggingFace token for pyannote gated models.\n"
        f"Original error: {exc}"
    )

# torch may or may not be available; whisperx needs it but we guard anyway
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Resolve project-level time utilities
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.time_utils import seconds_to_ts  # noqa: E402

logger = logging.getLogger(__name__)


class MeetingDiarizer:
    """WhisperX + pyannote speaker diarization pipeline.

    Attributes:
        hf_token:           HuggingFace access token (required for pyannote).
        whisperx_model_size: Whisper model size passed to whisperx.load_model().
        device:             'cpu' or 'cuda'.
        compute_type:       Float precision — 'float16' for GPU, 'int8' for CPU.
        min_speakers:       Minimum expected speaker count for pyannote.
        max_speakers:       Maximum expected speaker count for pyannote.
    """

    def __init__(
        self,
        hf_token: str,
        whisperx_model_size: str = "small",
        device: str = "cpu",
        compute_type: Optional[str] = None,
        min_speakers: int = 2,
        max_speakers: int = 20,
        batch_size: int = 16,
    ) -> None:
        """Initialize WhisperX + pyannote diarization pipeline.

        Args:
            hf_token:           HuggingFace token with access to pyannote models.
            whisperx_model_size: Whisper model size (tiny, base, small, medium, large-v2, etc.).
            device:             Compute device — 'cpu' or 'cuda'.
            compute_type:       Precision type. Defaults to 'float16' on CUDA, 'int8' on CPU.
            min_speakers:       Minimum number of speakers expected in the audio.
            max_speakers:       Maximum number of speakers expected in the audio.
            batch_size:         Batch size for WhisperX transcription.

        Raises:
            ImportError: If whisperx is not installed.
        """
        if not _WHISPERX_AVAILABLE:
            raise ImportError(_WHISPERX_IMPORT_ERROR)

        self.hf_token = hf_token
        self.whisperx_model_size = whisperx_model_size
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.batch_size = batch_size

        # Determine compute type based on device if not explicitly set
        if compute_type is not None:
            self.compute_type = compute_type
        elif device == "cuda" and torch is not None and torch.cuda.is_available():
            self.compute_type = "float16"
        else:
            self.compute_type = "int8"

        # Eagerly load the Whisper model so we fail fast on config errors
        logger.info(
            "Loading WhisperX model '%s' on %s (compute_type=%s)",
            whisperx_model_size, device, self.compute_type,
        )
        self._whisper_model = whisperx.load_model(
            whisperx_model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("WhisperX model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diarize(self, audio_path: str) -> list[dict]:
        """Run the full diarization pipeline on an audio file.

        Steps:
            1. WhisperX transcription (batched).
            2. WhisperX alignment for word-level timestamps.
            3. pyannote speaker diarization.
            4. WhisperX speaker assignment (merge diarization into aligned words).

        Args:
            audio_path: Path to a WAV / MP3 / FLAC audio file.

        Returns:
            List of segment dicts, each containing::

                {
                    "start": float,      # segment start in seconds
                    "end": float,        # segment end in seconds
                    "text": str,         # transcribed text
                    "speaker": str,      # e.g. "SPEAKER_00"
                    "words": list[dict], # word-level details
                }

        Raises:
            FileNotFoundError: If audio_path does not exist.
            ImportError:       If whisperx is not installed.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Starting diarization for: %s", audio_path)

        # --- Step 1: Transcribe -------------------------------------------
        logger.info("Step 1/4: Transcribing with WhisperX (%s)...", self.whisperx_model_size)
        audio = whisperx.load_audio(audio_path)
        result = self._whisper_model.transcribe(audio, batch_size=self.batch_size)
        detected_language = result.get("language", "en")
        logger.info("Transcription complete. Detected language: %s, segments: %d",
                     detected_language, len(result.get("segments", [])))

        # --- Step 2: Align ------------------------------------------------
        logger.info("Step 2/4: Aligning word-level timestamps...")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=self.device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        logger.info("Alignment complete. Word-aligned segments: %d", len(result.get("segments", [])))

        # Free alignment model memory
        del align_model
        if torch is not None:
            torch.cuda.empty_cache()

        # --- Step 3: Diarize ----------------------------------------------
        logger.info("Step 3/4: Running pyannote speaker diarization "
                     "(min_speakers=%d, max_speakers=%d)...",
                     self.min_speakers, self.max_speakers)
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=self.hf_token,
            device=self.device,
        )
        diarize_segments = diarize_model(
            audio,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )
        logger.info("Diarization complete.")

        # --- Step 4: Assign speakers to words/segments --------------------
        logger.info("Step 4/4: Assigning speakers to transcript segments...")
        result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = result.get("segments", [])
        logger.info("Speaker assignment complete. Final segments: %d", len(segments))

        # Normalize output: ensure every segment has a 'speaker' key
        for seg in segments:
            seg.setdefault("speaker", "UNKNOWN")
            seg.setdefault("words", [])

        return segments

    def segments_to_dataframe(self, segments: list[dict]) -> pd.DataFrame:
        """Convert diarized segments to the standard Meeting Process Twin event DataFrame.

        The output DataFrame has the same schema as the rest of the pipeline
        (timestamp, activity_name, source, details, original_text) plus an
        additional 'speaker' column.

        Args:
            segments: List of segment dicts from :meth:`diarize`.

        Returns:
            pd.DataFrame with columns:
                - timestamp: str (HH:MM:SS)
                - activity_name: str (defaults to 'Speech')
                - source: str ('diarization')
                - details: str (speaker label)
                - original_text: str (transcribed text)
                - speaker: str (e.g. 'SPEAKER_00')
                - start_seconds: float
                - end_seconds: float
        """
        rows = []
        for seg in segments:
            start_sec = seg.get("start", 0.0)
            end_sec = seg.get("end", start_sec)
            text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "UNKNOWN")

            rows.append({
                "timestamp": seconds_to_ts(int(start_sec)),
                "activity_name": "Speech",
                "source": "diarization",
                "details": f"Speaker: {speaker}",
                "original_text": text,
                "speaker": speaker,
                "start_seconds": round(start_sec, 2),
                "end_seconds": round(end_sec, 2),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            # Return empty DataFrame with the expected columns
            df = pd.DataFrame(columns=[
                "timestamp", "activity_name", "source", "details",
                "original_text", "speaker", "start_seconds", "end_seconds",
            ])
        return df

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def diarize_to_dataframe(self, audio_path: str) -> pd.DataFrame:
        """One-call convenience: diarize audio and return a DataFrame directly."""
        segments = self.diarize(audio_path)
        return self.segments_to_dataframe(segments)

    def __repr__(self) -> str:
        return (
            f"MeetingDiarizer(model={self.whisperx_model_size!r}, "
            f"device={self.device!r}, "
            f"speakers={self.min_speakers}-{self.max_speakers})"
        )
