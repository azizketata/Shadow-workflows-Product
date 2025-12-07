import os
import tempfile
import cv2
import mediapipe as mp
import pandas as pd
from moviepy import VideoFileClip
from openai import OpenAI
import time

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
            
            # Write audio to the temp file
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
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
                
                events.append({
                    'timestamp': timestamp_str,
                    'activity_name': 'Discussion',
                    'source': 'Audio',
                    'details': segment.text[:50] + "..." if len(segment.text) > 50 else segment.text,
                    'raw_seconds': start_time
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
        
        # 3. Combine and Sort
        all_events = audio_events + visual_events
        
        if not all_events:
            return pd.DataFrame(columns=['timestamp', 'activity_name', 'source', 'details'])
            
        df = pd.DataFrame(all_events)
        df = df.sort_values(by='raw_seconds').drop(columns=['raw_seconds']).reset_index(drop=True)
        
        return df

