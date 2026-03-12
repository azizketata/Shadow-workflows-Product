import os
import tempfile
import cv2
import numpy as np
import pandas as pd
from moviepy import VideoFileClip
from openai import OpenAI
import time

import spacy
import subprocess
import shutil
import re

# Lazy import for rtmlib (ONNX-based pose estimation, no protobuf dependency)
_RTMLIB_AVAILABLE = False
try:
    from rtmlib import Body
    _RTMLIB_AVAILABLE = True
except ImportError:
    pass

class VideoProcessor:
    def __init__(self, api_key, debug=False):
        """
        Initialize the VideoProcessor with OpenAI API key and visual detection backends.
        Uses rtmlib (RTMPose) for pose estimation and OpenCV MOG2 for motion detection.
        """
        self._api_key = api_key
        self._client = None  # Lazy-init: only created when API transcription is used
        self.debug = debug

        # --- Pose detection via rtmlib (replaces MediaPipe) ---
        self.body_detector = None
        if _RTMLIB_AVAILABLE:
            # Try GPU first, fall back to CPU
            for _device in ('cuda', 'cpu'):
                try:
                    self.body_detector = Body(
                        mode='lightweight',
                        backend='onnxruntime',
                        device=_device,
                    )
                    self._log(f"rtmlib Body detector initialized (RTMPose lightweight, ONNX, {_device})")
                    break
                except Exception as e:
                    if _device == 'cuda':
                        self._log(f"CUDA init failed, falling back to CPU: {e}")
                    else:
                        self._log(f"rtmlib init failed: {e}")
                        self.body_detector = None
        else:
            self._log("rtmlib not available — pose detection disabled")

        # --- Motion detection via OpenCV background subtraction ---
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False,
        )
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Resolve ffmpeg binary for moviepy/imageio
        self.ffmpeg_path = None
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            self.ffmpeg_path = get_ffmpeg_exe()
            os.environ["FFMPEG_BINARY"] = self.ffmpeg_path
            try:
                from moviepy.config import change_settings
                change_settings({"FFMPEG_BINARY": self.ffmpeg_path})
            except Exception:
                pass
        except Exception:
            self.ffmpeg_path = shutil.which("ffmpeg")
        
        self._log(f"FFmpeg path: {self.ffmpeg_path}")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if model not found, though it should be installed
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    @property
    def client(self):
        """Lazy-initialize OpenAI client only when needed (e.g., API transcription)."""
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _log(self, message):
        if self.debug:
            print(f"[VideoProcessor] {message}")

    def _format_timestamp(self, total_seconds):
        """Convert raw seconds to HH:MM:SS string."""
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
            
    def _filter_no_speech_segments(self, segments, threshold=0.6):
        """Filter out segments where Whisper's no_speech_prob exceeds threshold."""
        filtered = []
        skipped = 0
        for seg in segments:
            no_speech = seg.get("no_speech_prob", 0.0)
            if no_speech > threshold:
                skipped += 1
                continue
            filtered.append(seg)
        if skipped > 0:
            self._log(f"VAD filter: removed {skipped} silent/noise segments (no_speech_prob > {threshold})")
        return filtered

    def _match_phrase(self, text_lower, patterns):
        """Check if any of the given regex patterns match in text (word-boundary aware)."""
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
        return False

    def _extract_topic(self, doc, text):
        """Extract a topic phrase from spaCy doc for enriched 'Discussion: [topic]' labels."""
        # Try noun chunks first
        chunks = list(doc.noun_chunks)
        if chunks:
            meaningful = [
                chunk.text for chunk in chunks
                if chunk.root.pos_ != "PRON" and len(chunk.text) > 2
            ]
            if meaningful:
                topic = meaningful[0]
                words = topic.split()
                if len(words) > 5:
                    topic = " ".join(words[:5])
                return topic.strip()

        # Fallback: first 4 significant words (nouns/adjectives/proper nouns)
        significant = [
            token.text for token in doc
            if token.pos_ in ("NOUN", "PROPN", "ADJ") and not token.is_stop
        ]
        if significant:
            return " ".join(significant[:4])

        return None

    def extract_action_object(self, text):
        """
        Extract potential activities using NLP (Verb + Object).
        Returns a suggested Activity Name or None.
        """
        if not text:
            return None, None

        try:
            doc = self.nlp(text)
        except Exception:
            return "Discussion", text

        # 1. Keyword Mapping (Heuristics for council meeting terms)
        text_lower = text.lower()

        # Vote Results — must come BEFORE generic "motion" check
        if self._match_phrase(text_lower, [r'\bthe ayes have it\b', r'\bmotion carries\b', r'\bmotion passes\b']):
            return "Vote Result", text
        if self._match_phrase(text_lower, [r'\bmotion failed\b', r'\bmotion fails\b']):
            return "Vote Result", text

        # Seconds — require deliberate phrasing (avoids "second item" false positive)
        if self._match_phrase(text_lower, [r'\bi second\b', r'\bseconded\b', r'\bsecond the\b', r'\bsecond that\b']):
            return "Second Motion", text

        # Motions — require deliberate phrasing (avoids "move on" false positive)
        if self._match_phrase(text_lower, [
            r'\bi move\b', r'\bmake a motion\b', r'\bso moved\b',
            r'\bmotion to\b', r'\bmove to approve\b', r'\bmove to adopt\b'
        ]):
            return "Propose Motion", text
        # Generic "motion" only as standalone noun (not "move on", "move forward")
        if self._match_phrase(text_lower, [r'\bmotion\b']) and not self._match_phrase(text_lower, [
            r'\bmove on\b', r'\bmove forward\b', r'\bmoving on\b'
        ]):
            return "Propose Motion", text

        # Voting
        if self._match_phrase(text_lower, [r'\ball in favor\b', r'\bthose in favor\b']):
            return "Call for Vote", text
        if self._match_phrase(text_lower, [r'\bopposed\b', r'\bnays?\b']):
            return "Call for Vote", text
        if self._match_phrase(text_lower, [r'\broll call\b']):
            return "Roll Call Vote", text
        if self._match_phrase(text_lower, [r'\bvote\b']):
            return "Call for Vote", text

        # Adjournment / Recess
        if self._match_phrase(text_lower, [r'\badjourn']):
            return "Adjourn Meeting", text
        if self._match_phrase(text_lower, [r'\brecess\b']):
            return "Recess", text

        # Public Comment
        if self._match_phrase(text_lower, [r'\bpublic comment\b', r'\bpublic hearing\b']):
            return "Public Comment", text
        if self._match_phrase(text_lower, [r'\bopen the floor\b', r'\bfloor is open\b']):
            return "Open Public Comment", text
        if "close the" in text_lower and self._match_phrase(text_lower, [r'\bcomment\b', r'\bhearing\b']):
            return "Close Public Comment", text

        # Agenda items
        if self._match_phrase(text_lower, [r'\bagenda item\b', r'\bnext item\b', r'\bitem number\b']):
            return "Introduce Agenda Item", text
        if self._match_phrase(text_lower, [r'\bordinance\b']):
            return "Discuss Ordinance", text
        if self._match_phrase(text_lower, [r'\bresolution\b']):
            return "Discuss Resolution", text
        if self._match_phrase(text_lower, [r'\bapproval\b', r'\bapprove\b']):
            return "Request Approval", text

        # Presentations / Reports
        if self._match_phrase(text_lower, [r'\bpresent\b']) and self._match_phrase(text_lower, [r'\breport\b', r'\bupdate\b']):
            return "Staff Presentation", text
        if self._match_phrase(text_lower, [r'\bstaff report\b']):
            return "Staff Report", text
        if self._match_phrase(text_lower, [r'\bbudget\b']):
            return "Budget Discussion", text

        # 2. General NLP Extraction (Verb + Object with prepositional objects)
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                dobj = None
                pobj = None
                for child in token.children:
                    if child.dep_ == "dobj":
                        dobj = " ".join([t.text for t in child.subtree])
                    elif child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                pobj = f"{child.text} {' '.join(t.text for t in grandchild.subtree)}"
                                break
                if dobj:
                    label = f"{token.lemma_.capitalize()} {dobj.capitalize()}"
                    if pobj:
                        label += f" {pobj}"
                    return label, text
                elif pobj:
                    return f"{token.lemma_.capitalize()} {pobj.capitalize()}", text

        # 3. Topic-enriched fallback (instead of generic "Discussion")
        topic = self._extract_topic(doc, text)
        if topic:
            return f"Discussion: {topic}", text
        return "Discussion", text

    def extract_audio(self, video_path):
        """
        Extract audio from the video file using MoviePy.
        Returns the path to the temporary audio file.
        """
        try:
            self._log(f"Extracting audio from video: {video_path}")
            if not os.path.exists(video_path):
                self._log("Video path does not exist.")
                return None
            video = VideoFileClip(video_path)
            # Create a temp file for audio
            fd, audio_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            
            try:
                # Write audio to the temp file
                # verbose deprecated in recent moviepy versions, remove it
                video.audio.write_audiofile(audio_path, logger=None)
            finally:
                video.close()
            
            if os.path.exists(audio_path):
                audio_size = os.path.getsize(audio_path)
                self._log(f"Audio extracted: {audio_path} ({audio_size} bytes)")
            else:
                self._log("Audio extraction failed: output file missing.")
            
            return audio_path
        except Exception as e:
            self._log(f"Error extracting audio: {e}")
            return None

    def split_audio_into_chunks(self, audio_path, max_size_mb=20):
        """
        Split a large audio file into chunks under the specified size limit.
        Uses moviepy which is already installed and has ffmpeg bundled.
        
        Args:
            audio_path: Path to the original audio file
            max_size_mb: Maximum size per chunk in MB (default 20MB to safely stay under Whisper's 25MB limit)
            
        Returns:
            list: List of tuples (chunk_path, time_offset_seconds)
        """
        from moviepy import AudioFileClip
        import logging
        
        # Suppress moviepy's verbose logging
        logging.getLogger("moviepy").setLevel(logging.ERROR)
        
        chunks = []
        audio_clip = None
        
        try:
            if not audio_path or not os.path.exists(audio_path):
                self._log("split_audio_into_chunks: audio path missing.")
                return []

            # Get file size in bytes
            file_size = os.path.getsize(audio_path)
            max_size_bytes = max_size_mb * 1024 * 1024
            self._log(f"Audio file size: {file_size} bytes, max per chunk: {max_size_bytes} bytes")
            
            # If file is small enough, return it directly
            if file_size <= max_size_bytes:
                self._log("Audio file under size limit; skipping chunking.")
                return [(audio_path, 0)]
            
            # Load the audio file with moviepy
            audio_clip = AudioFileClip(audio_path)
            total_duration_seconds = audio_clip.duration
            self._log(f"Audio duration: {total_duration_seconds:.2f}s")
            
            # Estimate how many chunks we need (add extra margin)
            num_chunks = (file_size // max_size_bytes) + 2
            chunk_duration_seconds = total_duration_seconds / num_chunks
            self._log(f"Chunking into ~{num_chunks} chunks, target duration: {chunk_duration_seconds:.2f}s")
            
            # Split into chunks
            current_pos = 0
            chunk_index = 0
            
            while current_pos < total_duration_seconds:
                # Calculate end position
                end_pos = min(current_pos + chunk_duration_seconds, total_duration_seconds)
                
                # Save chunk to temp file
                fd, chunk_path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                
                # Prefer ffmpeg direct split for reliability
                if not self.ffmpeg_path:
                    self._log("FFmpeg not found; cannot split audio reliably.")
                    break

                cmd = [
                    self.ffmpeg_path,
                    "-y",
                    "-ss",
                    str(current_pos),
                    "-t",
                    str(end_pos - current_pos),
                    "-i",
                    audio_path,
                    "-acodec",
                    "libmp3lame",
                    "-b:a",
                    "128k",
                    chunk_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self._log(f"FFmpeg split failed (chunk {chunk_index}): {result.stderr.strip()}")
                    try:
                        os.remove(chunk_path)
                    except Exception:
                        pass
                    # Continue trying next chunk
                    current_pos = end_pos
                    chunk_index += 1
                    continue
                
                if os.path.exists(chunk_path):
                    chunk_size = os.path.getsize(chunk_path)
                    self._log(f"Chunk {chunk_index}: {chunk_path} ({chunk_size} bytes)")
                else:
                    self._log(f"Chunk {chunk_index}: file missing after write.")
                
                # Store chunk info with time offset in seconds
                chunks.append((chunk_path, current_pos))
                
                current_pos = end_pos
                chunk_index += 1
                
            self._log(f"Split audio into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            # Clean up any chunks created before the error
            for chunk_path, _ in chunks:
                if chunk_path != audio_path and os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except:
                        pass
            self._log(f"Error splitting audio: {e}")
            # Return empty to avoid oversized Whisper requests
            return []
        finally:
            # Always close the audio clip to release resources
            if audio_clip is not None:
                try:
                    audio_clip.close()
                except:
                    pass
    
    def _cleanup_chunk_files(self, chunks, original_audio_path):
        """
        Safely clean up all chunk files.
        
        Args:
            chunks: List of (chunk_path, offset) tuples
            original_audio_path: Path to the original audio file (always clean this up)
        """
        for chunk_path, _ in chunks:
            if chunk_path != original_audio_path and os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except Exception as e:
                    print(f"Warning: Could not remove chunk file {chunk_path}: {e}")
        
        # Always clean up original audio file
        if original_audio_path and os.path.exists(original_audio_path):
            try:
                os.remove(original_audio_path)
            except Exception as e:
                print(f"Warning: Could not remove audio file {original_audio_path}: {e}")

    def _process_transcript_segments(self, segments, time_offset):
        """Convert Whisper transcript segments to event dicts."""
        events = []
        for segment in segments:
            start_time = segment["start"] if isinstance(segment, dict) else segment.start
            text_content = (segment["text"] if isinstance(segment, dict) else segment.text).strip()
            if not text_content:
                continue
            activity_name, full_text = self.extract_action_object(text_content)
            if full_text:
                events.append({
                    'timestamp': self._format_timestamp(start_time + time_offset),
                    'activity_name': activity_name,
                    'source': 'Audio',
                    'details': full_text[:100] + "..." if len(full_text) > 100 else full_text,
                    'raw_seconds': start_time + time_offset,
                    'original_text': full_text,
                })
        return events

    def transcribe_audio_local(self, audio_path, progress_callback=None, model_size="base"):
        """
        Transcribe audio using the local openai-whisper model (no API cost).
        Falls back to API transcription if local whisper is unavailable.
        """
        try:
            import whisper as local_whisper
        except ImportError:
            self._log("Local whisper not installed; falling back to API.")
            return self.transcribe_audio(audio_path, progress_callback)

        events = []
        if not audio_path or not os.path.exists(audio_path):
            self._log("transcribe_audio_local: audio path missing.")
            return events

        try:
            if progress_callback:
                progress_callback(35, f"Loading local Whisper model ({model_size})...")
            self._log(f"Loading local Whisper model: {model_size}")
            import torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            self._log(f"Whisper device: {_device}")
            model = local_whisper.load_model(model_size, device=_device)

            if progress_callback:
                progress_callback(40, "Transcribing with local Whisper (this may take a while)...")
            self._log("Starting local Whisper transcription...")

            COUNCIL_PROMPT = (
                "City council meeting. Mayor, council members, staff, public attendees. "
                "Topics: motions, voting, public comment, budget, ordinances, resolutions."
            )
            result = model.transcribe(
                audio_path,
                verbose=False,
                initial_prompt=COUNCIL_PROMPT,
                language="en",
                word_timestamps=True,
            )
            segments = result.get("segments", [])
            # Filter out silent/noise segments using Whisper's no_speech_prob
            segments = self._filter_no_speech_segments(segments, threshold=0.6)
            self._log(f"Local Whisper produced {len(segments)} segments (after VAD filter)")
            events = self._process_transcript_segments(segments, time_offset=0)

            if progress_callback:
                progress_callback(70, f"Local transcription complete: {len(events)} events")
        except Exception as e:
            self._log(f"Local Whisper error: {e}")
            print(f"[VideoProcessor] Local Whisper failed: {e}")
        finally:
            # Free Whisper model from GPU memory
            try:
                del model
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

        return events

    def transcribe_audio(self, audio_path, progress_callback=None):
        """
        Transcribe audio using OpenAI Whisper API.
        Handles large files by splitting into chunks under 25MB.
        Returns a list of event dicts.
        """
        events = []
        chunks = []

        if not audio_path or not os.path.exists(audio_path):
            self._log("transcribe_audio: audio path missing or invalid.")
            return events

        try:
            # Split audio into chunks if needed
            if progress_callback:
                progress_callback(35, "Splitting audio into chunks...")

            chunks = self.split_audio_into_chunks(audio_path)
            self._log(f"Transcription chunks count: {len(chunks)}")
            if not chunks:
                self._log("No audio chunks available; skipping transcription.")
                return events

            total_chunks = len(chunks)
            for i, (chunk_path, time_offset) in enumerate(chunks):
                try:
                    if not os.path.exists(chunk_path):
                        self._log(f"Chunk missing: {chunk_path}")
                        continue
                    chunk_size = os.path.getsize(chunk_path)
                    if chunk_size < 1024:
                        self._log(f"Chunk too small, skipping: {chunk_path} ({chunk_size} bytes)")
                        continue

                    if progress_callback:
                        percent = 40 + int((i / total_chunks) * 30)
                        progress_callback(percent, f"Transcribing chunk {i+1}/{total_chunks}...")

                    with open(chunk_path, "rb") as audio_file:
                        transcript = self.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="verbose_json",
                            prompt=(
                                "This is a city council meeting recording. "
                                "Speakers include the Mayor, council members, city staff, and public attendees. "
                                "Topics include motions, voting, public comment, budget discussions, "
                                "ordinances, resolutions, and agenda items."
                            ),
                        )
                    events.extend(self._process_transcript_segments(transcript.segments, time_offset))

                except Exception as e:
                    self._log(f"Error transcribing chunk {chunk_path}: {e}")
                    continue

        except Exception as e:
            self._log(f"Error transcribing audio: {e}")
        finally:
            self._cleanup_chunk_files(chunks, audio_path)

        return events

    def process_visuals(self, video_path):
        """
        Process video frames for visual events using:
        - RTMPose (via rtmlib) for hand-raise / voting detection
        - OpenCV MOG2 background subtraction for speaker-activity / motion detection
        Returns a combined list of visual event dicts.
        """
        events = []
        # Force FFMPEG backend — Windows MSMF hangs on VP9 codec videos
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            self._log("Error opening video file for visual processing")
            return events

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            self._log("FPS invalid, defaulting to 30")
            fps = 30.0

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_duration_s = total_frames / fps if fps > 0 else 0

        # Adaptive sampling: 1-5 second intervals depending on video length
        sample_every_seconds = max(1, min(5, int(total_duration_s / 600)))
        sample_interval = max(1, int(fps * sample_every_seconds))
        self._log(
            f"Visual sampling: duration={total_duration_s:.0f}s, "
            f"sample_every={sample_every_seconds}s (every {sample_interval} frames)"
        )

        # Debounce trackers (last event timestamp in seconds)
        last_pose_event_sec = -999.0
        last_motion_event_sec = -999.0
        POSE_DEBOUNCE = 5.0    # seconds between hand-raise events
        MOTION_DEBOUNCE = 10.0  # seconds between motion events
        MOG2_WARMUP_SEC = 15.0  # skip motion detection until MOG2 has enough history

        import time as _time
        _visual_start_time = _time.time()

        last_progress_pct = -1
        total_samples = int(total_frames / sample_interval) if sample_interval > 0 else 0
        self._log(f"Starting frame loop: {int(total_frames)} total frames, {total_samples} samples (every {sample_interval} frames)")

        # Seek-based sampling: jump directly to sample frames instead of reading every frame
        for sample_idx in range(total_samples + 1):
            frame_num = sample_idx * sample_interval
            if frame_num >= total_frames:
                break
            current_seconds = frame_num / fps

            # Progress logging every 5%
            if total_samples > 0:
                pct = int(sample_idx / total_samples * 100) // 5 * 5
                if pct > last_progress_pct:
                    last_progress_pct = pct
                    elapsed_vis = _time.time() - _visual_start_time
                    self._log(f"Visual processing: {pct}% (sample {sample_idx}/{total_samples}, {current_seconds:.0f}s/{total_duration_s:.0f}s, {len(events)} events, {elapsed_vis:.0f}s elapsed)")

            # Seek to the target frame and read
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                self._log(f"Frame read failed at sample {sample_idx} (frame {frame_num}) — stopping")
                break

            # Feed frame to MOG2 for background model warmup (even when not emitting events)
            if current_seconds < MOG2_WARMUP_SEC:
                try:
                    small = cv2.resize(frame, (320, 240))
                    self.bg_subtractor.apply(small)
                except Exception:
                    pass

            # --- Pose detection (hand raises, standing, formal objection) ---
            pose_events = self._detect_pose_events(
                frame, current_seconds, last_pose_event_sec, POSE_DEBOUNCE
            )
            if pose_events:
                events.extend(pose_events)
                last_pose_event_sec = current_seconds

            # --- Motion detection (speaker activity) ---
            motion_event = self._detect_motion_events(
                frame, current_seconds, last_motion_event_sec, MOTION_DEBOUNCE
            )
            if motion_event:
                events.append(motion_event)
                last_motion_event_sec = current_seconds

        cap.release()
        self._log(f"Visual processing complete: {len(events)} events")
        return events

    def _detect_pose_events(self, frame, current_seconds, last_event_sec, debounce):
        """
        Detect parliamentary gestures using rtmlib RTMPose (COCO-17 keypoints).
        Detects: hand raises (voting), both hands raised (formal objection),
        group votes (multiple persons), and standing (speaker change).
        Returns a list of event dicts, or None.
        """
        if self.body_detector is None:
            return None

        # Debounce check
        if (current_seconds - last_event_sec) < debounce:
            return None

        try:
            keypoints, scores = self.body_detector(frame)
        except Exception as e:
            self._log(f"RTMPose error at {current_seconds:.1f}s: {e}")
            return None

        if keypoints is None or len(keypoints) == 0:
            return None

        MIN_KP_CONF = 0.3
        SHOULDER_MARGIN = 30   # pixels above shoulder to count as "raised"
        STANDING_MIN_DIST = 150  # pixels, hip-to-ankle vertical distance

        single_hand_persons = 0
        both_hands_persons = 0
        standing_persons = 0

        for person_idx in range(len(keypoints)):
            kp = keypoints[person_idx]   # shape (17, 2) — x, y
            sc = scores[person_idx]      # shape (17,)

            nose_y, nose_conf = kp[0][1], sc[0]
            lwrist_y, lwrist_conf = kp[9][1], sc[9]
            rwrist_y, rwrist_conf = kp[10][1], sc[10]
            lshoulder_y, lshoulder_conf = kp[5][1], sc[5]
            rshoulder_y, rshoulder_conf = kp[6][1], sc[6]

            # --- Hand raise detection ---
            if nose_conf >= MIN_KP_CONF:
                left_raised = (
                    lwrist_conf > MIN_KP_CONF
                    and lshoulder_conf > MIN_KP_CONF
                    and lwrist_y < nose_y
                    and lwrist_y < (lshoulder_y - SHOULDER_MARGIN)
                )
                right_raised = (
                    rwrist_conf > MIN_KP_CONF
                    and rshoulder_conf > MIN_KP_CONF
                    and rwrist_y < nose_y
                    and rwrist_y < (rshoulder_y - SHOULDER_MARGIN)
                )

                if left_raised and right_raised:
                    both_hands_persons += 1
                elif left_raised or right_raised:
                    single_hand_persons += 1

            # --- Standing detection (hip-to-ankle vertical distance) ---
            lhip_y, lhip_conf = kp[11][1], sc[11]
            rhip_y, rhip_conf = kp[12][1], sc[12]
            lankle_y, lankle_conf = kp[15][1], sc[15]
            rankle_y, rankle_conf = kp[16][1], sc[16]

            hip_y = None
            if lhip_conf > MIN_KP_CONF and rhip_conf > MIN_KP_CONF:
                hip_y = (lhip_y + rhip_y) / 2
            elif lhip_conf > MIN_KP_CONF:
                hip_y = lhip_y
            elif rhip_conf > MIN_KP_CONF:
                hip_y = rhip_y

            ankle_y = None
            if lankle_conf > MIN_KP_CONF and rankle_conf > MIN_KP_CONF:
                ankle_y = (lankle_y + rankle_y) / 2
            elif lankle_conf > MIN_KP_CONF:
                ankle_y = lankle_y
            elif rankle_conf > MIN_KP_CONF:
                ankle_y = rankle_y

            if hip_y is not None and ankle_y is not None:
                if (ankle_y - hip_y) > STANDING_MIN_DIST:
                    standing_persons += 1

        # --- Build event list ---
        events = []
        ts = self._format_timestamp(current_seconds)

        if both_hands_persons > 0:
            events.append({
                'timestamp': ts,
                'activity_name': 'Formal Objection',
                'source': 'Video',
                'details': f'Both hands raised (RTMPose) - {both_hands_persons} person(s)',
                'raw_seconds': current_seconds,
            })

        total_hands = single_hand_persons + both_hands_persons
        if single_hand_persons > 0:
            if total_hands > 1:
                activity = f"Group Vote ({total_hands} hands)"
            else:
                activity = "Voting"
            events.append({
                'timestamp': ts,
                'activity_name': activity,
                'source': 'Video',
                'details': f'Hand Raised (RTMPose) - {total_hands} person(s)',
                'raw_seconds': current_seconds,
                'hands_count': total_hands,
            })

        if standing_persons > 0:
            activity = "Standing" if standing_persons == 1 else f"Standing ({standing_persons} persons)"
            events.append({
                'timestamp': ts,
                'activity_name': activity,
                'source': 'Video',
                'details': f'Standing detected (RTMPose) - {standing_persons} person(s)',
                'raw_seconds': current_seconds,
            })

        return events if events else None

    def _detect_motion_events(self, frame, current_seconds, last_event_sec, debounce):
        """
        Detect significant motion / speaker activity using OpenCV MOG2 background subtraction.
        Uses morphological opening to remove noise pixels, then checks if >15% are foreground.
        Returns an event dict or None.
        """
        # Skip during MOG2 warmup period (first 15 seconds produce false positives)
        if current_seconds < 15.0:
            return None

        # Debounce check
        if (current_seconds - last_event_sec) < debounce:
            return None

        try:
            # Resize to 320x240 for speed
            small = cv2.resize(frame, (320, 240))
            fg_mask = self.bg_subtractor.apply(small)
            # Morphological opening to remove salt-and-pepper noise pixels
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._morph_kernel)
            fg_ratio = np.count_nonzero(fg_mask) / fg_mask.size
        except Exception as e:
            self._log(f"Motion detection error at {current_seconds:.1f}s: {e}")
            return None

        MOTION_THRESHOLD = 0.15  # 15% foreground pixels (raised from 8% to reduce false positives)
        if fg_ratio > MOTION_THRESHOLD:
            return {
                'timestamp': self._format_timestamp(current_seconds),
                'activity_name': 'Speaker Activity',
                'source': 'Video',
                'details': f'Motion detected ({fg_ratio:.1%} foreground)',
                'raw_seconds': current_seconds,
            }

        return None

    def _link_visual_to_nearest_audio(self, visual_events, audio_events, max_gap_seconds=15):
        """Link each visual event to the nearest audio event within ±max_gap_seconds.

        Adds 'nearest_audio_activity' field for downstream agenda mapping.
        """
        if not audio_events or not visual_events:
            return visual_events

        audio_seconds = np.array([e['raw_seconds'] for e in audio_events])

        for vis in visual_events:
            vis_sec = vis['raw_seconds']
            diffs = np.abs(audio_seconds - vis_sec)
            min_idx = int(np.argmin(diffs))
            if diffs[min_idx] <= max_gap_seconds:
                vis['nearest_audio_activity'] = audio_events[min_idx].get('activity_name', '')
            else:
                vis['nearest_audio_activity'] = None

        return visual_events

    def fuse_events(self, audio_events, visual_events):
        """
        Fuse audio and visual events based on temporal proximity and logic.

        Fusion rules:
        1. Audio "Vote" + Visual "Voting" within ±5s → "Confirmed Vote" (Fused)
        2. Speech-related audio + Visual "Speaker Activity" within ±10s → source "Audio+Motion"
        3. Unused visual events are appended as standalone
        """
        fused_events = []

        df_audio = pd.DataFrame(audio_events)
        df_visual = pd.DataFrame(visual_events)

        if df_audio.empty and df_visual.empty:
            return []
        if df_visual.empty:
            return audio_events
        if df_audio.empty:
            return visual_events

        used_visual_indices = set()

        # Separate visual events by type for targeted matching
        voting_mask = df_visual['activity_name'].str.contains('Voting|Group Vote|Formal Objection', na=False)
        motion_mask = df_visual['activity_name'] == 'Speaker Activity'
        df_voting = df_visual[voting_mask]
        df_motion = df_visual[motion_mask]

        # Only speech-related activities can be enriched with motion (avoids enriching procedural events)
        MOTION_ENRICHABLE = {
            'Public Comment', 'Open Public Comment', 'Close Public Comment',
            'Discussion', 'Staff Presentation', 'Staff Report',
            'Budget Discussion', 'Discuss Ordinance', 'Discuss Resolution',
        }

        for _, audio_row in df_audio.iterrows():
            audio_sec = audio_row['raw_seconds']
            is_vote_context = "Vote" in str(audio_row['activity_name'])
            fused = False

            # Rule 1: Audio vote + Visual hand raise → Confirmed Vote
            if is_vote_context and not df_voting.empty:
                nearby_votes = df_voting[
                    (df_voting['raw_seconds'] >= audio_sec - 5) &
                    (df_voting['raw_seconds'] <= audio_sec + 5)
                ]
                if not nearby_votes.empty:
                    fused_events.append({
                        'timestamp': audio_row['timestamp'],
                        'activity_name': 'Confirmed Vote',
                        'source': 'Fused (Audio+Video)',
                        'details': f"Audio: {audio_row['details']} | Visual: {len(nearby_votes)} hands detected",
                        'raw_seconds': audio_sec,
                    })
                    for idx in nearby_votes.index:
                        used_visual_indices.add(idx)
                    fused = True

            # Rule 2: Speech-related audio + nearby motion → Audio+Motion enrichment
            if not fused and not df_motion.empty:
                activity = str(audio_row['activity_name'])
                is_enrichable = (
                    activity in MOTION_ENRICHABLE
                    or activity.startswith("Discussion:")
                )
                if is_enrichable:
                    nearby_motion = df_motion[
                        (df_motion['raw_seconds'] >= audio_sec - 10) &
                        (df_motion['raw_seconds'] <= audio_sec + 10)
                    ]
                    if not nearby_motion.empty:
                        event = audio_row.to_dict()
                        event['source'] = 'Audio+Motion'
                        fused_events.append(event)
                        for idx in nearby_motion.index:
                            used_visual_indices.add(idx)
                        fused = True

            if not fused:
                fused_events.append(audio_row.to_dict())

        # Append unused visual events as standalone
        for idx, vis_row in df_visual.iterrows():
            if idx not in used_visual_indices:
                fused_events.append(vis_row.to_dict())

        return fused_events

    def _cluster_events(self, events, gap_seconds=5):
        """Group near-simultaneous events and keep the highest-confidence per cluster.

        Priority: Fused > Audio+Motion > Audio > Video; specific label > generic Discussion.
        """
        if not events or len(events) <= 1:
            return events

        SOURCE_PRIORITY = {
            'Fused (Audio+Video)': 4,
            'Audio+Motion': 3,
            'Audio': 2,
            'Video': 1,
        }

        def event_priority(e):
            source_score = SOURCE_PRIORITY.get(e.get('source', 'Audio'), 2)
            name = str(e.get('activity_name', ''))
            name_score = 0 if name == 'Discussion' else 1
            return (source_score, name_score)

        sorted_events = sorted(events, key=lambda e: e.get('raw_seconds', 0))

        clusters = [[sorted_events[0]]]
        for evt in sorted_events[1:]:
            prev_time = clusters[-1][-1].get('raw_seconds', 0)
            curr_time = evt.get('raw_seconds', 0)
            if (curr_time - prev_time) <= gap_seconds:
                clusters[-1].append(evt)
            else:
                clusters.append([evt])

        result = [max(cluster, key=event_priority) for cluster in clusters]
        deduplicated = len(events) - len(result)
        if deduplicated > 0:
            self._log(f"Temporal clustering: {len(events)} -> {len(result)} (-{deduplicated} duplicates)")

        return result

    def process_video(self, video_path, use_local_whisper=False, local_whisper_model="base",
                       progress_callback=None, return_separate=False):
        """
        Main method to process both audio and visuals.

        Args:
            use_local_whisper: If True, use local openai-whisper (no API cost).
            local_whisper_model: Model size for local whisper ('tiny','base','small','medium','large').
            progress_callback: Optional callable(percent: int, text: str) for progress updates.
                               If None, progress is silently ignored (headless mode).
            return_separate: If True, return (audio_events, visual_events) lists instead of fused DataFrame.
                             Useful for separate caching in test_pipeline.py.

        Returns:
            pd.DataFrame (default) or tuple(list, list) if return_separate=True.
        """
        def _progress(percent, text=""):
            if progress_callback:
                progress_callback(percent, text)

        # 1. Process Audio
        _progress(10, "Extracting audio from video...")
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            self._log("Audio extraction returned None; skipping transcription.")

        _progress(30, "Preparing audio for transcription...")

        if audio_path:
            if use_local_whisper:
                audio_events = self.transcribe_audio_local(
                    audio_path,
                    progress_callback=progress_callback,
                    model_size=local_whisper_model,
                )
            else:
                audio_events = self.transcribe_audio(
                    audio_path, progress_callback=progress_callback
                )
        else:
            audio_events = []
        self._log(f"Audio events count: {len(audio_events)}")

        # 2. Process Visuals (rtmlib pose + OpenCV motion — degrades gracefully)
        _progress(70, "Analyzing visual gestures...")
        try:
            import threading
            vis_result = [None]
            vis_error = [None]
            def _run_visuals():
                try:
                    vis_result[0] = self.process_visuals(video_path)
                except Exception as e:
                    vis_error[0] = e
            vis_thread = threading.Thread(target=_run_visuals, daemon=True)
            vis_thread.start()
            # Timeout: 20 min max for frame analysis
            vis_thread.join(timeout=600)  # 10 min max
            if vis_thread.is_alive():
                self._log("WARNING: Visual processing timed out (10 min) — skipping")
                print(f"[VideoProcessor] WARNING: visual processing timed out — skipping")
                visual_events = []
            elif vis_error[0]:
                raise vis_error[0]
            else:
                visual_events = vis_result[0] or []
            self._log(f"Visual events count: {len(visual_events)}")
        except Exception as e:
            self._log(f"Visual processing failed (skipping): {e}")
            print(f"[VideoProcessor] WARNING: visual processing skipped — {e}")
            visual_events = []

        # Link visual events to nearest audio events for downstream mapping
        visual_events = self._link_visual_to_nearest_audio(visual_events, audio_events)

        # Return raw lists for separate caching if requested
        if return_separate:
            return audio_events, visual_events

        # 3. Fuse + temporal clustering
        _progress(90, "Fusing audio and visual events...")
        fused_list = self.fuse_events(audio_events, visual_events)
        fused_list = self._cluster_events(fused_list, gap_seconds=5)

        if not fused_list:
            _progress(100, "Analysis complete!")
            return pd.DataFrame(columns=['timestamp', 'activity_name', 'source', 'details'])

        df = pd.DataFrame(fused_list)
        df = df.sort_values(by='raw_seconds').drop(columns=['raw_seconds']).reset_index(drop=True)

        _progress(100, "Analysis complete!")
        return df

