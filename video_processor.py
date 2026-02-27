import os
import tempfile
import cv2
import mediapipe as mp
import pandas as pd
from moviepy import VideoFileClip
from openai import OpenAI
import time

import spacy
import subprocess
import shutil
import re

class VideoProcessor:
    def __init__(self, api_key, debug=False):
        """
        Initialize the VideoProcessor with OpenAI API key and MediaPipe Pose.
        """
        self.client = OpenAI(api_key=api_key)
        self.mp_pose = mp.solutions.pose
        self.debug = debug
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
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
    
    def _log(self, message):
        if self.debug:
            print(f"[VideoProcessor] {message}")

    def _format_timestamp(self, total_seconds):
        """Convert raw seconds to HH:MM:SS string."""
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = int(total_seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
            
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
        # Motions
        # Vote Results — must come BEFORE generic "motion" check to avoid false match
        if "the ayes have it" in text_lower or "motion carries" in text_lower or "motion passes" in text_lower:
            return "Vote Result", text
        if "motion failed" in text_lower or "motion fails" in text_lower:
            return "Vote Result", text
        # Seconds — check before generic "motion" to avoid "second the motion" → Propose Motion
        if "i second" in text_lower or "seconded" in text_lower:
            return "Second Motion", text
        if "second" in text_lower and ("motion" in text_lower or "that" in text_lower):
            return "Second Motion", text
        # Motions
        if "i move" in text_lower or "make a motion" in text_lower or "so moved" in text_lower:
            return "Propose Motion", text
        if "motion" in text_lower and ("introduce" in text_lower or "present" in text_lower):
            return "Propose Motion", text
        if ("motion" in text_lower or "move" in text_lower) and "second" not in text_lower:
            return "Propose Motion", text
        # Voting
        if "all in favor" in text_lower or "those in favor" in text_lower:
            return "Call for Vote", text
        if "opposed" in text_lower or "nay" in text_lower or "nays" in text_lower:
            return "Call for Vote", text
        if "roll call" in text_lower:
            return "Roll Call Vote", text
        if "vote" in text_lower:
            return "Call for Vote", text
        # Adjournment / Recess
        if "adjourn" in text_lower:
            return "Adjourn Meeting", text
        if "recess" in text_lower:
            return "Recess", text
        # Public Comment
        if "public comment" in text_lower or "public hearing" in text_lower:
            return "Public Comment", text
        if "open the floor" in text_lower or "floor is open" in text_lower:
            return "Open Public Comment", text
        if "close the" in text_lower and ("comment" in text_lower or "hearing" in text_lower):
            return "Close Public Comment", text
        # Agenda items
        if "agenda item" in text_lower or "next item" in text_lower or "item number" in text_lower:
            return "Introduce Agenda Item", text
        if "ordinance" in text_lower:
            return "Discuss Ordinance", text
        if "resolution" in text_lower:
            return "Discuss Resolution", text
        if "approval" in text_lower or "approve" in text_lower:
            return "Request Approval", text
        # Presentations / Reports
        if "present" in text_lower and ("report" in text_lower or "update" in text_lower):
            return "Staff Presentation", text
        if "staff report" in text_lower:
            return "Staff Report", text
        if "budget" in text_lower:
            return "Budget Discussion", text
            
        # 2. General NLP Extraction (Verb + Object)
        for token in doc:
            if token.pos_ == "VERB":
                # Find direct object
                for child in token.children:
                    if child.dep_ == "dobj":
                        return f"{token.lemma_.capitalize()} {child.text.capitalize()}", text
                        
        return "Discussion", text # Default fallback

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
            model = local_whisper.load_model(model_size)

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
                word_timestamps=False,
            )
            segments = result.get("segments", [])
            self._log(f"Local Whisper produced {len(segments)} segments")
            events = self._process_transcript_segments(segments, time_offset=0)

            if progress_callback:
                progress_callback(70, f"Local transcription complete: {len(events)} events")
        except Exception as e:
            self._log(f"Local Whisper error: {e}")
            print(f"[VideoProcessor] Local Whisper failed: {e}")
        finally:
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
        Process video frames to detect gestures (raised hands) using MediaPipe Pose.
        Returns a list of 'Voting' events.
        """
        events = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error opening video file")
            return events

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print("Warning: FPS is 0 or invalid, defaulting to 30 FPS")
            fps = 30.0

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_duration_s = total_frames / fps if fps > 0 else 0

        # For long videos, sample at most every 5 seconds to keep processing fast.
        # For short videos keep per-second resolution (every fps frames).
        sample_every_seconds = max(1, min(5, int(total_duration_s / 600)))
        sample_interval = max(1, int(fps * sample_every_seconds))
        self._log(
            f"Visual sampling: duration={total_duration_s:.0f}s, "
            f"sample_every={sample_every_seconds}s (every {sample_interval} frames)"
        )

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Adaptive sample interval
            if frame_count % sample_interval == 0:
                try:
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(image_rgb)
                except Exception as e:
                    print(f"[VideoProcessor] MediaPipe frame error at frame {frame_count}: {e}")
                    cap.release()
                    return events  # return what we have so far
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Get relevant landmarks
                    nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                    left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    
                    # Check if hand is raised (Y coordinate is smaller when higher in image)
                    # We assume 'raised' means wrist is significantly above the nose
                    # Adding a small threshold or just strict comparison
                    is_left_raised = left_wrist.y < nose.y
                    is_right_raised = right_wrist.y < nose.y
                    
                    if is_left_raised or is_right_raised:
                        # Calculate timestamp
                        current_seconds = frame_count / fps

                        # Debounce: skip if we already logged a voting event within 5 seconds
                        if events and events[-1]['activity_name'] == 'Voting' and \
                                (current_seconds - events[-1]['raw_seconds']) < 5:
                            frame_count += 1
                            continue

                        events.append({
                            'timestamp': self._format_timestamp(current_seconds),
                            'activity_name': 'Voting',
                            'source': 'Video',
                            'details': 'Hand Raised',
                            'raw_seconds': current_seconds
                        })

            frame_count += 1
            
        cap.release()
        return events

    def fuse_events(self, audio_events, visual_events):
        """
        Fuse audio and visual events based on temporal proximity and logic.
        """
        fused_events = []
        
        # Convert to DataFrames for easier handling if they aren't already
        df_audio = pd.DataFrame(audio_events)
        df_visual = pd.DataFrame(visual_events)
        
        if df_audio.empty and df_visual.empty:
            return []
            
        if df_visual.empty:
            return audio_events
            
        if df_audio.empty:
            return visual_events

        # Track used visual events to avoid duplication
        used_visual_indices = set()

        # Iterate through audio events (primary stream)
        for _, audio_row in df_audio.iterrows():
            # Check for visual confirmation (Voting)
            # Logic: If Audio says "Vote" and Visual has "Voting" within +/- 5 seconds
            
            is_vote_context = "Vote" in audio_row['activity_name']
            
            # Find nearby visual events
            nearby_visuals = df_visual[
                (df_visual['raw_seconds'] >= audio_row['raw_seconds'] - 5) & 
                (df_visual['raw_seconds'] <= audio_row['raw_seconds'] + 5)
            ]
            
            if is_vote_context and not nearby_visuals.empty:
                # FUSION: Confirmed Vote
                fused_events.append({
                    'timestamp': audio_row['timestamp'],
                    'activity_name': 'Confirmed Vote', # Fused Event Name
                    'source': 'Fused (Audio+Video)',
                    'details': f"Audio: {audio_row['details']} | Visual: {len(nearby_visuals)} hands detected",
                    'raw_seconds': audio_row['raw_seconds']
                })
                # Mark these visual events as consumed
                for idx in nearby_visuals.index:
                    used_visual_indices.add(idx)
            else:
                # No fusion, keep original audio event
                fused_events.append(audio_row.to_dict())
                
        # Add visual events that weren't used in fusion
        # For independent gestures
        for idx, vis_row in df_visual.iterrows():
             if idx not in used_visual_indices:
                 fused_events.append(vis_row.to_dict())
             
        return fused_events

    def process_video(self, video_path, use_local_whisper=False, local_whisper_model="base"):
        """
        Main method to process both audio and visuals.
        Returns a Pandas DataFrame.

        Args:
            use_local_whisper: If True, use local openai-whisper (no API cost).
            local_whisper_model: Model size for local whisper ('tiny','base','small','medium','large').
        """
        # Create a progress bar
        import streamlit as st
        progress_bar = st.progress(0, text="Starting video analysis...")

        # 1. Process Audio
        progress_bar.progress(10, text="Extracting audio from video...")
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            self._log("Audio extraction returned None; skipping transcription.")

        progress_bar.progress(30, text="Preparing audio for transcription...")

        def update_transcription_progress(percent, text):
            progress_bar.progress(percent, text=text)

        if audio_path:
            if use_local_whisper:
                audio_events = self.transcribe_audio_local(
                    audio_path,
                    progress_callback=update_transcription_progress,
                    model_size=local_whisper_model,
                )
            else:
                audio_events = self.transcribe_audio(
                    audio_path, progress_callback=update_transcription_progress
                )
        else:
            audio_events = []
        self._log(f"Audio events count: {len(audio_events)}")
        
        # 2. Process Visuals (optional — degrades gracefully on MediaPipe errors)
        progress_bar.progress(70, text="Analyzing visual gestures...")
        try:
            visual_events = self.process_visuals(video_path)
            self._log(f"Visual events count: {len(visual_events)}")
        except Exception as e:
            self._log(f"Visual processing failed (skipping): {e}")
            print(f"[VideoProcessor] WARNING: visual processing skipped — {e}")
            visual_events = []
        
        # 3. Fuse
        progress_bar.progress(90, text="Fusing audio and visual events...")
        fused_list = self.fuse_events(audio_events, visual_events)
        
        if not fused_list:
            progress_bar.empty()
            return pd.DataFrame(columns=['timestamp', 'activity_name', 'source', 'details'])
            
        df = pd.DataFrame(fused_list)
        df = df.sort_values(by='raw_seconds').drop(columns=['raw_seconds']).reset_index(drop=True)
        
        progress_bar.progress(100, text="Analysis complete!")
        time.sleep(1) # Let user see completion
        progress_bar.empty()
        
        return df

