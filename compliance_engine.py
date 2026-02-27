import pm4py
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import json
import urllib.request
import urllib.error
from typing import List, Optional
from difflib import get_close_matches
import concurrent.futures
import threading
import openai
from keyword_rules import keyword_map, condense_window_text, summarize_visual_events

class ComplianceEngine:
    def __init__(self, sbert_model_name='all-MiniLM-L6-v2'):
        """
        Initialize the ComplianceEngine with SBERT model.

        Args:
            sbert_model_name: SBERT model to use. Options:
                - 'all-MiniLM-L6-v2' (fast, good coverage)
                - 'all-mpnet-base-v2' (slower, higher precision)
        """
        self.sbert_model_name = sbert_model_name
        self._llm_semaphore = threading.Semaphore(5)  # max 5 concurrent LLM calls
        try:
            self.model = SentenceTransformer(sbert_model_name)
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

    def _build_classification_prompt(self, agenda_tasks, window_events, visual_context=""):
        agenda_list = "\n".join(f"  - {t}" for t in agenda_tasks) if isinstance(agenda_tasks, list) else str(agenda_tasks)

        visual_section = ""
        if visual_context:
            visual_section = (
                "\nVISUAL EVIDENCE (from video analysis):\n"
                f"{visual_context}\n"
                "Consider visual cues: hand raises often indicate voting, "
                "motion spikes indicate active discussion or speaker changes.\n\n"
            )

        return (
            "You are an expert city council meeting analyst. Classify a window of meeting events "
            "into a single activity label using the rules below.\n\n"
            "FORMAL AGENDA TASKS (the planned activities for this meeting — prefer these):\n"
            f"{agenda_list}\n\n"
            "CLASSIFICATION RULES (in priority order):\n"
            "1. FORMAL FIRST: If the events fit ANY of the formal agenda tasks above — even loosely — "
            "return that EXACT task label. Err on the side of formal labels.\n"
            "2. SHADOW (only if no agenda task fits): If events clearly represent an off-agenda informal "
            "pattern (e.g. unscheduled break, private sidebar, pre-meeting chatter), return: "
            "Shadow: <short descriptive label>\n"
            "3. NOISE: Only if the window is truly unintelligible filler (silence, crosstalk, 'um', 'yeah' "
            "only), return: Noise\n\n"
            "CRITICAL GUIDANCE FOR COUNCIL MEETINGS:\n"
            "- Citizens speaking on ANY topic during public comment = 'Public Comment' (formal)\n"
            "- 'Come to order', opening statements = 'Call to Order'\n"
            "- Reading out names of council members = 'Roll Call'\n"
            "- Saying the Pledge = 'Pledge of Allegiance'\n"
            "- Approving the meeting agenda = 'Approval of Agenda'\n"
            "- Approving last meeting notes = 'Approval of Minutes'\n"
            "- Department head giving an update = 'City Manager Report' or 'Finance Department Update'\n"
            "- Council members reporting on their work = 'Council Member Reports'\n"
            "- Hearing from the public on a specific ordinance = 'Open Public Hearing'\n"
            "- Motion, second, vote on consent items = 'Approve Consent Agenda Items'\n"
            "- Discussing a pending ordinance = 'Review Pending Ordinances'\n"
            "- Introducing a new ordinance = 'Introduce New Ordinance'\n"
            "- Budget discussion = 'Budget Amendments Discussion'\n"
            "- Vote on resolution = 'Resolution for Capital Improvements'\n"
            "- Contract vote = 'Contract Approval'\n"
            "- City attorney speaking = 'City Attorney Report'\n"
            "- Announcing future meetings = 'Scheduling and Announcements'\n"
            "- Meeting closing = 'Adjourn Meeting'\n\n"
            "EXAMPLES:\n"
            "Events: citizens speaking about police accountability during public comment "
            "-> Return: Public Comment\n"
            "Events: 'move to approve', 'second the motion', 'all in favor' "
            "-> Return: Approve Consent Agenda Items\n"
            "Events: council president reading out council member names "
            "-> Return: Roll Call\n"
            "Events: city attorney discussing pending litigation "
            "-> Return: City Attorney Report\n"
            "Events: council member talking about a community event they attended "
            "-> Return: Council Member Reports\n"
            "Events: two council members whispering, off-mic chatter between agenda items "
            "-> Return: Shadow: Informal Side Conversation\n"
            "Events: 'um', 'yeah', 'okay', background noise only "
            "-> Return: Noise\n\n"
            f"{visual_section}"
            "IMPORTANT: Return ONLY the label string. No explanation, no JSON, no quotes around it.\n\n"
            "MEETING EVENTS TO CLASSIFY:\n"
            f"{window_events}"
        )

    def _ollama_classify_window(self, model, agenda_tasks, window_events, visual_context="", timeout=60):
        prompt = self._build_classification_prompt(agenda_tasks, window_events, visual_context)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise meeting process analyst. Always respond with only the classification label."},
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

    def _openai_classify_window(self, model, agenda_tasks, window_events, api_key, visual_context="", timeout=60):
        if not api_key:
            return "Noise"

        prompt = self._build_classification_prompt(agenda_tasks, window_events, visual_context)

        try:
            client = openai.OpenAI(api_key=api_key, timeout=timeout)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise meeting process analyst. Always respond with only the classification label."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI classification error: {e}")
            return "Noise"

    def _fuzzy_match_label(self, label, agenda_tasks, cutoff=0.6):
        """Fuzzy-match an LLM-returned label to the closest agenda task.

        Catches near-misses like 'Budget Amendment' vs 'Budget Amendments Discussion'.
        """
        if not label or not agenda_tasks:
            return label
        stripped = label.strip()
        # Exact match
        if stripped in agenda_tasks:
            return stripped
        # Case-insensitive exact match
        lower_map = {t.lower(): t for t in agenda_tasks}
        if stripped.lower() in lower_map:
            return lower_map[stripped.lower()]
        # Fuzzy match
        matches = get_close_matches(stripped, agenda_tasks, n=1, cutoff=cutoff)
        if matches:
            return matches[0]
        return stripped

    def abstract_events_df(
        self,
        df,
        agenda_tasks,
        window_seconds=60,
        shadow_min_ratio=0.15,
        model="mistral",
        overlap_ratio=0.5,
        min_events_per_window=2,
        min_label_support=1,
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

            # Phase A: Collect all windows, separate cached from uncached
            uncached_windows = []
            cached_results = []

            for start_t, end_t in windows:
                window_df = case_df[(case_df["__seconds"] >= start_t) & (case_df["__seconds"] < end_t)]
                if window_df.empty:
                    continue
                window_event_counts.append(len(window_df))
                if len(window_df) < min_events_per_window:
                    label_counts["Noise"] += 1
                    continue

                # Condensed window text (suggestion n) + visual context (suggestion o)
                condensed_text = condense_window_text(window_df, start_t, end_t)
                visual_ctx = summarize_visual_events(window_df)

                cache_key = (
                    backend,
                    model if backend == "ollama" else openai_model,
                    case_id,
                    start_t,
                    end_t,
                )
                if cache_key in cache:
                    cached_results.append((case_id, start_t, end_t, cache[cache_key], window_df))
                else:
                    uncached_windows.append((case_id, start_t, end_t, window_df, condensed_text, visual_ctx, cache_key))

            # Phase B: Parallel LLM dispatch for uncached windows (suggestion p)
            def _classify_window(item):
                _cid, _st, _et, _wdf, _text, _vis_ctx, _ckey = item
                with self._llm_semaphore:
                    if backend == "ollama":
                        _label = self._ollama_classify_window(
                            model=model,
                            agenda_tasks=agenda_tasks,
                            window_events=_text,
                            visual_context=_vis_ctx,
                        )
                    else:
                        _label = self._openai_classify_window(
                            model=openai_model,
                            agenda_tasks=agenda_tasks,
                            window_events=_text,
                            api_key=api_key,
                            visual_context=_vis_ctx,
                            timeout=openai_timeout,
                        )
                cache[_ckey] = _label
                return (_cid, _st, _et, _label, _wdf)

            all_results = list(cached_results)

            if uncached_windows:
                max_workers = min(5, len(uncached_windows))
                debug(f"Dispatching {len(uncached_windows)} LLM calls with {max_workers} workers")
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_classify_window, item): item for item in uncached_windows}
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            all_results.append(future.result())
                        except Exception as e:
                            debug(f"LLM call failed: {e}")
                            item = futures[future]
                            all_results.append((item[0], item[1], item[2], "Noise", item[3]))

            # Phase C: Process results with fuzzy matching (suggestion q)
            for _cid, _st, _et, label, _wdf in all_results:
                if not label:
                    label = "Noise"

                # Fuzzy-match to agenda tasks for near-misses
                if not label.strip().lower().startswith("shadow:") and label.strip().lower() != "noise":
                    label = self._fuzzy_match_label(label, agenda_tasks)

                if label.strip().lower() == "noise":
                    label_counts["Noise"] += 1
                    continue

                if label.strip().lower().startswith("shadow:"):
                    label_counts["Shadow"] += 1
                else:
                    label_counts["Formal"] += 1

                aggregated_rows.append(
                    {
                        "timestamp": self._seconds_to_timestamp(_st),
                        "activity_name": label.strip(),
                        "source": "Abstracted",
                        "details": f"Aggregated {len(_wdf)} events",
                        "case_id": _cid,
                        "__seconds": _st,
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

    def map_events_to_agenda(self, video_events, agenda_activities, threshold=0.35):
        """
        Map extracted video events to agenda activities using a hybrid approach:
        1. Keyword rules (high-confidence, fast) — from shared keyword_rules module
        2. SBERT semantic similarity (batch-encoded, enriched with original_text)

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

        target_col = 'activity_name' if 'activity_name' in mapped_df.columns else 'concept:name'
        unique_labels = mapped_df[target_col].unique().tolist()

        # --- Pass 1: Keyword rules for high-confidence matches ---
        keyword_matches = {}
        for label in unique_labels:
            sample_rows = mapped_df[mapped_df[target_col] == label]
            if not sample_rows.empty:
                sample = sample_rows.iloc[0]
                km = keyword_map(sample.to_dict(), agenda_activities)
                if km:
                    keyword_matches[label] = km

        # --- Pass 2: SBERT for remaining unmatched labels ---
        sbert_labels = [l for l in unique_labels if l not in keyword_matches]
        label_map = dict(keyword_matches)

        if sbert_labels:
            # Pre-compute embeddings for agenda activities
            agenda_embeddings = self.model.encode(agenda_activities, convert_to_tensor=True)

            # Enrich labels with original_text for richer semantic matching (suggestion s)
            enriched_texts = []
            for label in sbert_labels:
                sample_rows = mapped_df[mapped_df[target_col] == label]
                if not sample_rows.empty and 'original_text' in mapped_df.columns:
                    orig = str(sample_rows.iloc[0].get('original_text', '')).strip()
                    if orig and orig != 'nan':
                        enriched_texts.append(f"{label}: {orig[:100]}")
                    else:
                        details = str(sample_rows.iloc[0].get('details', '')).strip()
                        if details and details != 'nan':
                            enriched_texts.append(f"{label}: {details[:100]}")
                        else:
                            enriched_texts.append(label)
                else:
                    enriched_texts.append(label)

            # Batch-encode all SBERT labels at once (suggestion r) — ~10x faster
            label_embeddings = self.model.encode(enriched_texts, convert_to_tensor=True)
            scores_matrix = util.cos_sim(label_embeddings, agenda_embeddings)

            for i, label in enumerate(sbert_labels):
                scores = scores_matrix[i]
                best_idx = int(scores.argmax())
                best_score = float(scores[best_idx])

                if best_score >= threshold:
                    label_map[label] = agenda_activities[best_idx]
                else:
                    label_map[label] = f"Deviation: {label}"

        mapped_df['mapped_activity'] = mapped_df[target_col].map(label_map)

        # Update the main activity column for PM4Py compliance checking
        mapped_df['concept:name'] = mapped_df['mapped_activity']

        return mapped_df

    def calculate_fitness(self, reference_bpmn, log_df):
        """
        Calculate fitness using both token-based replay and alignment-based methods.

        Args:
            reference_bpmn: The reference BPMN model.
            log_df (pd.DataFrame): The mapped event log.

        Returns:
            dict: {
                'score': float (token-based replay),
                'alignment_score': float (alignment-based, more precise with parallel gateways),
                'alignments': list
            }
        """
        if reference_bpmn is None or log_df is None or log_df.empty:
            return {'score': 0.0, 'alignment_score': 0.0, 'alignments': []}

        try:
            # Convert BPMN to Petri Net for conformance checking
            net, initial_marking, final_marking = pm4py.convert_to_petri_net(reference_bpmn)

            # Token-based replay (fast)
            fitness = pm4py.fitness_token_based_replay(log_df, net, initial_marking, final_marking)

            # Alignment-based (more precise, especially with parallel gateways)
            alignments = pm4py.conformance_diagnostics_alignments(log_df, net, initial_marking, final_marking)

            # Compute alignment-based fitness score
            alignment_fitness = 0.0
            if alignments:
                total_cost = sum(a.get("cost", 0) for a in alignments)
                total_possible = 0
                for a in alignments:
                    align_list = a.get("alignment", [])
                    total_possible += len(align_list)
                if total_possible > 0:
                    alignment_fitness = 1.0 - (total_cost / total_possible)

            return {
                'score': fitness['log_fitness'],
                'alignment_score': max(0.0, alignment_fitness),
                'alignments': alignments,
            }

        except Exception as e:
            print(f"Error calculating fitness: {e}")
            return {'score': 0.0, 'alignment_score': 0.0, 'alignments': []}

