"""Speaker diarization research module for Meeting Process Twin.

Provides speaker attribution, turn-taking analysis, and social network
discovery from meeting audio using WhisperX + pyannote pipelines.

Submodules:
    diarizer       -- MeetingDiarizer: WhisperX transcription + pyannote speaker segmentation
    speaker_events -- Merge speaker labels into existing event DataFrames
    social_network -- pm4py-based handover and working-together network discovery
    turn_taking    -- Speaker equity, interruption detection, turn-length metrics
"""
