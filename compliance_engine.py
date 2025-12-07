import pm4py
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st

class ComplianceEngine:
    def __init__(self):
        """
        Initialize the ComplianceEngine with SBERT model.
        """
        # Load SBERT model for semantic similarity
        # 'all-MiniLM-L6-v2' is fast and efficient
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading SBERT model: {e}")
            self.model = None

    def map_events_to_agenda(self, video_events, agenda_activities, threshold=0.5):
        """
        Map extracted video events to agenda activities using semantic similarity.
        
        Args:
            video_events (pd.DataFrame): DataFrame containing video events.
            agenda_activities (list): List of strings representing agenda items.
            threshold (float): Similarity threshold to accept a match.
            
        Returns:
            pd.DataFrame: A new DataFrame with mapped 'concept:name'.
        """
        if self.model is None or video_events.empty or not agenda_activities:
            return video_events

        mapped_df = video_events.copy()
        
        # Pre-compute embeddings for agenda activities
        agenda_embeddings = self.model.encode(agenda_activities, convert_to_tensor=True)
        
        def get_best_match(row):
            # Use the dynamically determined column name
            event_label = row[target_col]
            
            # Encode event label
            event_embedding = self.model.encode(event_label, convert_to_tensor=True)
            
            # Compute cosine similarities
            cosine_scores = util.cos_sim(event_embedding, agenda_embeddings)[0]
            
            # Find best match
            best_score_idx = cosine_scores.argmax()
            best_score = cosine_scores[best_score_idx]
            
            if best_score >= threshold:
                return agenda_activities[best_score_idx]
            else:
                return f"Deviation: {event_label}" # Or keep original name

        # Apply mapping
        # We assume column 'activity_name' exists from Phase 2
        # If we are working with the 'log_df' from convert_to_event_log, it has 'concept:name'
        
        target_col = 'activity_name' if 'activity_name' in mapped_df.columns else 'concept:name'
        
        # Optimize: Mapping unique labels only instead of every row for speed
        unique_labels = mapped_df[target_col].unique()
        label_map = {label: get_best_match({target_col: label}) for label in unique_labels}
        
        mapped_df['mapped_activity'] = mapped_df[target_col].map(label_map)
        
        # Update the main activity column for PM4Py compliance checking
        mapped_df['concept:name'] = mapped_df['mapped_activity']
        
        return mapped_df

    def calculate_fitness(self, reference_bpmn, log_df):
        """
        Calculate fitness between the reference BPMN model and the event log.
        
        Args:
            reference_bpmn (pm4py.objects.bpmn.obj.BPMN): The reference model.
            log_df (pd.DataFrame): The mapped event log.
            
        Returns:
            float: Fitness score (0.0 - 1.0)
        """
        if reference_bpmn is None or log_df is None or log_df.empty:
            return 0.0
            
        try:
            # Convert BPMN to Petri Net for conformance checking
            net, initial_marking, final_marking = pm4py.convert_to_petri_net(reference_bpmn)
            
            # Calculate fitness using token-based replay (fast)
            fitness = pm4py.fitness_token_based_replay(log_df, net, initial_marking, final_marking)
            
            # Extract log fitness
            return fitness['log_fitness']
            
        except Exception as e:
            print(f"Error calculating fitness: {e}")
            return 0.0

