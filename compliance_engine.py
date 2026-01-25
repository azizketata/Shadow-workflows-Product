import pm4py
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import json
import urllib.request
import urllib.error
from typing import List, Optional
import openai

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

    def _parse_timestamp_to_seconds(self, t_str):
        if pd.isna(t_str):
            return 0
        if isinstance(t_str, (int, float)):
            return int(t_str)
        parts = str(t_str).split(":")
        try:
            if len(parts) == 2:
                m, s = map(int, parts)
                return m * 60 + s
            if len(parts) == 3:
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
        except Exception:
            return 0
        return 0

    def _seconds_to_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _ollama_classify_window(self, model, agenda_tasks, window_events, timeout=60):
        prompt = (
            "You are abstracting micro-events from a meeting into higher-level activities.\n"
            "Known formal agenda tasks:\n"
            f"{agenda_tasks}\n\n"
            "For these events, return ONE of:\n"
            "- A matching formal agenda task\n"
            "- A high-level \"Shadow Workflow\" label (recurrent informal pattern)\n"
            "- \"Noise\" (discard)\n"
            "If Shadow, use format: \"Shadow: <Label>\".\n"
            "Return only the label.\n\n"
            "Events:\n"
            f"{window_events}"
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise classification assistant for meeting events."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "").strip()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            print(f"Ollama classification error: {e}")
            return "Noise"

    def _ollama_available(self, timeout=3):
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _openai_classify_window(self, model, agenda_tasks, window_events, api_key, timeout=60):
        if not api_key:
            return "Noise"

        prompt = (
            "You are abstracting micro-events from a meeting into higher-level activities.\n"
            "Known formal agenda tasks:\n"
            f"{agenda_tasks}\n\n"
            "For these events, return ONE of:\n"
            "- A matching formal agenda task\n"
            "- A high-level \"Shadow Workflow\" label (recurrent informal pattern)\n"
            "- \"Noise\" (discard)\n"
            "If Shadow, use format: \"Shadow: <Label>\".\n"
            "Return only the label.\n\n"
            "Events:\n"
            f"{window_events}"
        )

        try:
            client = openai.OpenAI(api_key=api_key, timeout=timeout)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise classification assistant for meeting events."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI classification error: {e}")
            return "Noise"

    def abstract_events_df(
        self,
        df,
        agenda_tasks,
        window_seconds=60,
        shadow_min_ratio=0.15,
        model="mistral",
        overlap_ratio=0.5,
        min_events_per_window=5,
        min_label_support=2,
        api_key=None,
        openai_model="gpt-4o-mini",
        openai_timeout=60,
        max_windows_per_run=100,
        cache=None,
        debug_callback=None,
    ):
        def debug(msg):
            if debug_callback:
                debug_callback(msg)

        if df is None or df.empty:
            debug("Abstraction skipped: input dataframe is empty.")
            return df

        if not agenda_tasks:
            debug("Abstraction skipped: no agenda tasks provided.")
            return df

        working_df = df.copy()
        if "case_id" not in working_df.columns:
            working_df["case_id"] = "Meeting_1"

        if "raw_seconds" in working_df.columns:
            working_df["__seconds"] = working_df["raw_seconds"].astype(int)
        else:
            working_df["__seconds"] = working_df["timestamp"].apply(self._parse_timestamp_to_seconds)

        working_df = working_df.sort_values(["case_id", "__seconds"]).reset_index(drop=True)
        step_seconds = max(1, int(window_seconds * (1 - overlap_ratio)))
        if cache is None:
            cache = {}

        ollama_available = self._ollama_available()
        backend = "ollama" if ollama_available else "openai"
        debug(
            f"Abstraction config -> window_seconds={window_seconds}, overlap_ratio={overlap_ratio}, "
            f"step_seconds={step_seconds}, min_events_per_window={min_events_per_window}, "
            f"min_label_support={min_label_support}, backend={backend}, "
            f"ollama_model={model}, openai_model={openai_model}, "
            f"cases={working_df['case_id'].nunique()}, rows={len(working_df)}"
        )
        if backend == "openai" and not api_key:
            debug("Abstraction fallback -> OpenAI selected but api_key is missing; will return Noise.")

        aggregated_rows = []
        label_counts = {"Noise": 0, "Shadow": 0, "Formal": 0}
        window_event_counts = []
        for case_id, case_df in working_df.groupby("case_id"):
            if case_df.empty:
                continue
            min_t = int(case_df["__seconds"].min())
            max_t = int(case_df["__seconds"].max())
            if max_t == min_t:
                windows = [(min_t, min_t + window_seconds)]
            else:
                windows = [(t, t + window_seconds) for t in range(min_t, max_t + 1, step_seconds)]
            if len(windows) > max_windows_per_run:
                stride = max(1, len(windows) // max_windows_per_run)
                windows = windows[::stride][:max_windows_per_run]
                debug(f"Window cap applied -> stride={stride}, capped_windows={len(windows)}")
            debug(f"Case {case_id}: windows={len(windows)}, min_t={min_t}, max_t={max_t}")

            for start_t, end_t in windows:
                window_df = case_df[(case_df["__seconds"] >= start_t) & (case_df["__seconds"] < end_t)]
                if window_df.empty:
                    continue
                window_event_counts.append(len(window_df))
                if len(window_df) < min_events_per_window:
                    label_counts["Noise"] += 1
                    continue

                window_events = []
                for _, row in window_df.iterrows():
                    activity = row.get("activity_name", "")
                    source = row.get("source", "")
                    details = row.get("details", "")
                    timestamp = row.get("timestamp", "")
                    window_events.append(f"- [{timestamp}] {activity} ({source}): {details}")

                cache_key = (
                    backend,
                    model if backend == "ollama" else openai_model,
                    case_id,
                    start_t,
                    end_t,
                )
                if cache_key in cache:
                    label = cache[cache_key]
                else:
                    if backend == "ollama":
                        label = self._ollama_classify_window(
                            model=model,
                            agenda_tasks=agenda_tasks,
                            window_events="\n".join(window_events),
                        )
                    else:
                        label = self._openai_classify_window(
                            model=openai_model,
                            agenda_tasks=agenda_tasks,
                            window_events="\n".join(window_events),
                            api_key=api_key,
                            timeout=openai_timeout,
                        )
                    cache[cache_key] = label

                if not label:
                    label = "Noise"

                if label.strip().lower() == "noise":
                    label_counts["Noise"] += 1
                    continue

                if label.strip().lower().startswith("shadow:"):
                    label_counts["Shadow"] += 1
                else:
                    label_counts["Formal"] += 1

                aggregated_rows.append(
                    {
                        "timestamp": self._seconds_to_timestamp(start_t),
                        "activity_name": label.strip(),
                        "source": "Abstracted",
                        "details": f"Aggregated {len(window_df)} events",
                        "case_id": case_id,
                        "__seconds": start_t,
                    }
                )

        debug(
            "Abstraction results -> "
            f"kept={len(aggregated_rows)}, noise={label_counts['Noise']}, "
            f"shadow={label_counts['Shadow']}, formal={label_counts['Formal']}"
        )
        if window_event_counts:
            debug(
                "Window event stats -> "
                f"min={min(window_event_counts)}, max={max(window_event_counts)}, "
                f"avg={sum(window_event_counts) / len(window_event_counts):.2f}"
            )

        if not aggregated_rows:
            debug("Abstraction produced no rows after noise filtering.")
            return pd.DataFrame(columns=["timestamp", "activity_name", "source", "details", "case_id"])

        aggregated_df = pd.DataFrame(aggregated_rows).sort_values(["case_id", "__seconds"]).reset_index(drop=True)

        shadow_rows = aggregated_df[aggregated_df["activity_name"].str.startswith("Shadow:", na=False)]
        unique_cases = aggregated_df["case_id"].nunique()
        if unique_cases > 1 and not shadow_rows.empty:
            shadow_counts = shadow_rows.groupby("activity_name")["case_id"].nunique()
            shadow_ratios = shadow_counts / unique_cases
            keep_shadows = set(shadow_ratios[shadow_ratios >= shadow_min_ratio].index.tolist())
            debug(
                f"Shadow filter -> unique_cases={unique_cases}, "
                f"candidates={len(shadow_counts)}, kept={len(keep_shadows)}"
            )
            aggregated_df = aggregated_df[
                (~aggregated_df["activity_name"].str.startswith("Shadow:", na=False))
                | (aggregated_df["activity_name"].isin(keep_shadows))
            ]
        elif unique_cases <= 1 and not shadow_rows.empty:
            debug("Shadow filter skipped (single case).")

        if min_label_support > 1 and not aggregated_df.empty:
            label_counts_total = aggregated_df["activity_name"].value_counts()
            keep_labels = set(label_counts_total[label_counts_total >= min_label_support].index.tolist())
            debug(
                f"Label support filter -> unique_labels={len(label_counts_total)}, kept={len(keep_labels)}"
            )
            aggregated_df = aggregated_df[aggregated_df["activity_name"].isin(keep_labels)]

        if not aggregated_df.empty:
            # Collapse consecutive duplicate labels within each case to reduce spaghetti
            aggregated_df["__prev_label"] = aggregated_df.groupby("case_id")["activity_name"].shift(1)
            aggregated_df = aggregated_df[aggregated_df["activity_name"] != aggregated_df["__prev_label"]]
            aggregated_df = aggregated_df.drop(columns=["__prev_label"], errors="ignore")

        aggregated_df = aggregated_df.drop(columns=["__seconds"], errors="ignore")
        return aggregated_df

    def abstract_events(self, raw_log_path, agenda_tasks, window_seconds=60, shadow_min_ratio=0.15, model="mistral"):
        if raw_log_path.lower().endswith(".csv"):
            df = pd.read_csv(raw_log_path)
        elif raw_log_path.lower().endswith(".json"):
            df = pd.read_json(raw_log_path)
        else:
            raise ValueError("Unsupported log format. Use CSV or JSON.")

        return self.abstract_events_df(
            df=df,
            agenda_tasks=agenda_tasks,
            window_seconds=window_seconds,
            shadow_min_ratio=shadow_min_ratio,
            model=model,
        )

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
        Calculate fitness and return detailed alignments for visualization.
        
        Args:
            reference_bpmn (pm4py.objects.bpmn.obj.BPMN): The reference model.
            log_df (pd.DataFrame): The mapped event log.
            
        Returns:
            dict: {
                'score': float,
                'alignments': list
            }
        """
        if reference_bpmn is None or log_df is None or log_df.empty:
            return {'score': 0.0, 'alignments': []}
            
        try:
            # Convert BPMN to Petri Net for conformance checking
            net, initial_marking, final_marking = pm4py.convert_to_petri_net(reference_bpmn)
            
            # Calculate fitness using token-based replay (fast)
            fitness = pm4py.fitness_token_based_replay(log_df, net, initial_marking, final_marking)
            
            # Calculate Alignments for visualization
            alignments = pm4py.conformance_diagnostics_alignments(log_df, net, initial_marking, final_marking)
            
            return {
                'score': fitness['log_fitness'],
                'alignments': alignments
            }
            
        except Exception as e:
            print(f"Error calculating fitness: {e}")
            return {'score': 0.0, 'alignments': []}

