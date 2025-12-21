import os
import tempfile
import cv2
import mediapipe as mp
import pandas as pd
from moviepy import VideoFileClip
from openai import OpenAI
import time

import spacy
import re

class VideoProcessor:
    def __init__(self, api_key):
        """
        Initialize the VideoProcessor with OpenAI API key and MediaPipe Pose.
        """
        self.client = OpenAI(api_key=api_key)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if model not found, though it should be installed
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
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
        
        # 1. Keyword Mapping (Heuristics for common meeting terms)
        text_lower = text.lower()
        if "move" in text_lower or "motion" in text_lower:
            return "Propose Motion", text
        if "second" in text_lower:
            return "Second Motion", text
        if "vote" in text_lower or "all in favor" in text_lower or "opposed" in text_lower:
            return "Call for Vote", text
        if "adjourn" in text_lower:
            return "Adjourn Meeting", text
            
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
                
            return audio_path
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio using OpenAI Whisper API.
        Returns a list of 'Discussion' events.
        """
        events = []
        if not audio_path or not os.path.exists(audio_path):
            return events

        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="verbose_json"
                )
            
            # Process segments
            for segment in transcript.segments:
                start_time = segment.start
                # Format timestamp as MM:SS
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                timestamp_str = f"{minutes:02d}:{seconds:02d}"
                
                text_content = segment.text.strip()
                if not text_content:
                    continue
                    
                activity_name, full_text = self.extract_action_object(text_content)
                
                # Double check to ensure we have valid text before appending
                if full_text:
                    events.append({
                        'timestamp': timestamp_str,
                        'activity_name': activity_name,
                        'source': 'Audio',
                        'details': full_text[:100] + "..." if len(full_text) > 100 else full_text,
                        'raw_seconds': start_time,
                        'original_text': full_text
                    })
                
        except Exception as e:
            print(f"Error transcribing audio: {e}")
        finally:
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
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
            
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every 30th frame
            if frame_count % 30 == 0:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(image_rgb)
                
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
                        minutes = int(current_seconds // 60)
                        seconds = int(current_seconds % 60)
                        timestamp_str = f"{minutes:02d}:{seconds:02d}"
                        
                        # Avoid spamming events - simple logic: check if we just added one recently?
                        # For now, per requirements: "Map this to a 'Voting' event"
                        # We might want to group them later, but let's stick to raw detection first
                        # or simple de-duplication if it's the same second.
                        
                        events.append({
                            'timestamp': timestamp_str,
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

    def process_video(self, video_path):
        """
        Main method to process both audio and visuals.
        Returns a Pandas DataFrame.
        """
        # 1. Process Audio
        audio_path = self.extract_audio(video_path)
        audio_events = self.transcribe_audio(audio_path)
        
        # 2. Process Visuals
        visual_events = self.process_visuals(video_path)
        
        # 3. Fuse
        fused_list = self.fuse_events(audio_events, visual_events)
        
        if not fused_list:
            return pd.DataFrame(columns=['timestamp', 'activity_name', 'source', 'details'])
            
        df = pd.DataFrame(fused_list)
        df = df.sort_values(by='raw_seconds').drop(columns=['raw_seconds']).reset_index(drop=True)
        
        return df

