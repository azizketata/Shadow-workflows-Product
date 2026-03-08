"""Process deviance classifier for shadow workflow events.

Uses rules first (fast, deterministic), then optionally LLM for ambiguous cases.
Can cross-reference with Declare violation results from research.declare.

Typical usage:
    classifier = DevianceClassifier(api_key="sk-...")
    classified = classifier.classify(shadow_events_df)
    summary = classifier.generate_deviance_summary(classified)
"""

from __future__ import annotations

import logging
from typing import Optional

import openai
import pandas as pd

from research.deviance.deviance_taxonomy import (
    CATEGORY_DESCRIPTIONS,
    DevianceCategory,
    classify_by_rules,
    get_severity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template for LLM-based deviance classification
# ---------------------------------------------------------------------------
_LLM_CLASSIFICATION_PROMPT = """\
You are an expert in civic meeting governance and Robert's Rules of Order.
Classify the following shadow workflow event into EXACTLY ONE of these categories:

Categories:
- BENIGN_FLEXIBILITY: Normal adaptive behavior (breaks, informal greetings, social interaction)
- PROCEDURAL_VIOLATION: Robert's Rules violations (voting without motion, speaking out of turn, no quorum)
- EFFICIENCY_GAIN: Streamlining behavior (combining items, waiving readings, expediting)
- INNOVATION: Novel deliberation approaches (workshops, straw polls, community input)
- EXTERNAL_DISRUPTION: Technical issues, emergencies, audience disruptions

Event activity: {activity_name}
Event details: {details}
Event original text: {original_text}

{declare_context}

Respond with ONLY the category name (e.g., BENIGN_FLEXIBILITY). No explanation."""


def _build_declare_context(declare_violations: list[dict] | None) -> str:
    """Build additional LLM context from Declare constraint violations."""
    if not declare_violations:
        return ""

    lines = ["Related Declare constraint violations detected in this meeting:"]
    for v in declare_violations[:5]:  # Limit to 5 most relevant
        constraint = v.get("constraint", "unknown")
        description = v.get("description", "")
        lines.append(f"  - {constraint}: {description}")

    lines.append(
        "\nConsider these violations when classifying. Events related to "
        "procedural violations should be classified as PROCEDURAL_VIOLATION."
    )
    return "\n".join(lines)


class DevianceClassifier:
    """Hybrid rules + LLM classifier for meeting process deviance.

    Operates in two modes:
    - Rules-only (default, no API key): Fast deterministic classification
      using keyword patterns from deviance_taxonomy.
    - Rules + LLM (with API key): Rules first, then LLM fallback for
      events classified as UNKNOWN by rules.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        llm_timeout: int = 30,
    ):
        """Initialize the deviance classifier.

        Args:
            api_key: OpenAI API key. If None, operates in rules-only mode.
            model: OpenAI model name for LLM fallback classification.
            llm_timeout: Timeout in seconds for each LLM call.
        """
        self.api_key = api_key
        self.model = model
        self.llm_timeout = llm_timeout
        self._client: openai.OpenAI | None = None

        if api_key:
            try:
                self._client = openai.OpenAI(api_key=api_key, timeout=llm_timeout)
            except Exception as exc:
                logger.warning("Failed to initialize OpenAI client: %s", exc)
                self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        shadow_events: pd.DataFrame,
        declare_violations: list[dict] | None = None,
    ) -> pd.DataFrame:
        """Classify each shadow event into a DevianceCategory.

        Adds two columns to the returned DataFrame:
        - deviance_category: The DevianceCategory value string (e.g. "benign")
        - deviance_rationale: How the classification was determined
            ("rule:<pattern>" or "llm" or "unknown")

        Only processes rows whose activity_name starts with "Shadow:" or
        whose mapped_activity starts with "Deviation:". Non-shadow rows
        are left with category="N/A" and rationale="formal_event".

        Args:
            shadow_events: DataFrame with at minimum an 'activity_name' column.
                Expected columns: timestamp, activity_name, source, details,
                original_text, mapped_activity.
            declare_violations: Optional list of dicts from Declare conformance
                checking, each with 'constraint' and 'description' keys.

        Returns:
            A copy of the input DataFrame with 'deviance_category' and
            'deviance_rationale' columns added.
        """
        if shadow_events is None or shadow_events.empty:
            return shadow_events

        df = shadow_events.copy()
        df["deviance_category"] = "N/A"
        df["deviance_rationale"] = "formal_event"

        # Identify shadow/deviation rows
        shadow_mask = self._get_shadow_mask(df)

        if not shadow_mask.any():
            logger.info("No shadow events found; skipping deviance classification.")
            return df

        logger.info(
            "Classifying %d shadow events (%d total rows).",
            shadow_mask.sum(),
            len(df),
        )

        # Phase 1: Rule-based classification
        declare_ctx = _build_declare_context(declare_violations)
        unknown_indices = []

        for idx in df.index[shadow_mask]:
            row = df.loc[idx]
            activity = str(row.get("activity_name", ""))
            details = str(row.get("details", ""))
            original_text = str(row.get("original_text", ""))
            combined_text = f"{details} {original_text}".strip()

            category = classify_by_rules(activity, combined_text)

            if category == DevianceCategory.UNKNOWN:
                unknown_indices.append(idx)
            else:
                df.at[idx, "deviance_category"] = category.value
                df.at[idx, "deviance_rationale"] = f"rule:{category.name}"

        logger.info(
            "Rule-based: %d classified, %d unknown.",
            shadow_mask.sum() - len(unknown_indices),
            len(unknown_indices),
        )

        # Phase 2: LLM fallback for UNKNOWN events
        if unknown_indices and self._client is not None:
            llm_classified = 0
            for idx in unknown_indices:
                row = df.loc[idx]
                category = self._llm_classify(
                    activity_name=str(row.get("activity_name", "")),
                    details=str(row.get("details", "")),
                    original_text=str(row.get("original_text", "")),
                    declare_context=declare_ctx,
                )
                df.at[idx, "deviance_category"] = category.value
                df.at[idx, "deviance_rationale"] = "llm"
                llm_classified += 1

            logger.info("LLM fallback: %d classified.", llm_classified)
        elif unknown_indices:
            # No LLM available -- mark as UNKNOWN
            for idx in unknown_indices:
                df.at[idx, "deviance_category"] = DevianceCategory.UNKNOWN.value
                df.at[idx, "deviance_rationale"] = "unknown:no_llm"

        return df

    def generate_deviance_summary(self, classified_df: pd.DataFrame) -> dict:
        """Generate a summary of deviance classification results.

        Args:
            classified_df: DataFrame with 'deviance_category' column, as
                returned by classify().

        Returns:
            Dictionary containing:
            - category_distribution: dict mapping category name to count
            - total_shadow_events: int
            - total_formal_events: int
            - severity_score: float (0-100 scale, weighted by severity)
            - max_possible_severity: float
            - severity_percentage: float (0-100)
            - top_violations: list of dicts with activity details for
                PROCEDURAL_VIOLATION events
            - recommendations: list of actionable recommendation strings
            - classification_method: dict mapping method to count
        """
        if classified_df is None or classified_df.empty:
            return self._empty_summary()

        shadow_mask = classified_df["deviance_category"] != "N/A"
        shadow_df = classified_df[shadow_mask]
        formal_count = int((~shadow_mask).sum())

        if shadow_df.empty:
            summary = self._empty_summary()
            summary["total_formal_events"] = formal_count
            return summary

        # Category distribution
        cat_dist = shadow_df["deviance_category"].value_counts().to_dict()

        # Severity computation
        total_severity = 0
        max_severity = 0
        max_weight = max(
            w for w in [get_severity(c) for c in DevianceCategory] if w > 0
        )

        for _, row in shadow_df.iterrows():
            cat_val = row["deviance_category"]
            try:
                cat = DevianceCategory(cat_val)
            except ValueError:
                cat = DevianceCategory.UNKNOWN
            total_severity += get_severity(cat)
            max_severity += max_weight

        severity_pct = (total_severity / max_severity * 100) if max_severity > 0 else 0.0

        # Top violations
        violations = shadow_df[
            shadow_df["deviance_category"] == DevianceCategory.PROCEDURAL_VIOLATION.value
        ]
        top_violations = []
        for _, row in violations.iterrows():
            top_violations.append({
                "timestamp": row.get("timestamp", ""),
                "activity_name": row.get("activity_name", ""),
                "details": row.get("details", ""),
                "rationale": row.get("deviance_rationale", ""),
            })

        # Classification method distribution
        method_dist = shadow_df["deviance_rationale"].value_counts().to_dict()

        # Generate recommendations
        recommendations = self._generate_recommendations(cat_dist, len(shadow_df))

        return {
            "category_distribution": cat_dist,
            "total_shadow_events": int(len(shadow_df)),
            "total_formal_events": formal_count,
            "severity_score": round(total_severity, 2),
            "max_possible_severity": round(max_severity, 2),
            "severity_percentage": round(severity_pct, 2),
            "top_violations": top_violations,
            "recommendations": recommendations,
            "classification_method": method_dist,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_shadow_mask(df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask identifying shadow / deviation rows.

        Only events whose activity_name starts with "Shadow:" are considered
        shadow events. Events with formal activity names (Roll Call, Budget
        Discussion, etc.) that happen to have a "Deviation:" mapped_activity
        are excluded — those are formal activities that simply didn't match
        a specific agenda item in SBERT mapping, not true shadow workflows.
        """
        mask = pd.Series(False, index=df.index)
        if "activity_name" in df.columns:
            mask |= df["activity_name"].astype(str).str.startswith("Shadow:")
        return mask

    def _llm_classify(
        self,
        activity_name: str,
        details: str,
        original_text: str,
        declare_context: str,
    ) -> DevianceCategory:
        """Classify a single event using the OpenAI LLM."""
        if self._client is None:
            return DevianceCategory.UNKNOWN

        prompt = _LLM_CLASSIFICATION_PROMPT.format(
            activity_name=activity_name,
            details=details[:300],
            original_text=original_text[:300],
            declare_context=declare_context,
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a civic meeting governance expert. "
                            "Classify process deviance precisely."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            label = response.choices[0].message.content.strip().upper()
            return self._parse_llm_label(label)

        except Exception as exc:
            logger.warning("LLM classification failed: %s", exc)
            return DevianceCategory.UNKNOWN

    @staticmethod
    def _parse_llm_label(label: str) -> DevianceCategory:
        """Parse an LLM response string into a DevianceCategory."""
        # Try exact match first
        for cat in DevianceCategory:
            if cat.name == label:
                return cat

        # Try partial / fuzzy match
        label_lower = label.lower()
        if "violation" in label_lower or "procedural" in label_lower:
            return DevianceCategory.PROCEDURAL_VIOLATION
        if "benign" in label_lower or "flexibility" in label_lower:
            return DevianceCategory.BENIGN_FLEXIBILITY
        if "efficien" in label_lower or "gain" in label_lower:
            return DevianceCategory.EFFICIENCY_GAIN
        if "innovat" in label_lower:
            return DevianceCategory.INNOVATION
        if "disrupt" in label_lower or "external" in label_lower:
            return DevianceCategory.EXTERNAL_DISRUPTION

        return DevianceCategory.UNKNOWN

    @staticmethod
    def _generate_recommendations(
        category_distribution: dict, total_shadows: int
    ) -> list[str]:
        """Generate actionable recommendations based on deviance patterns."""
        recommendations: list[str] = []

        violation_count = category_distribution.get(
            DevianceCategory.PROCEDURAL_VIOLATION.value, 0
        )
        disruption_count = category_distribution.get(
            DevianceCategory.EXTERNAL_DISRUPTION.value, 0
        )
        benign_count = category_distribution.get(
            DevianceCategory.BENIGN_FLEXIBILITY.value, 0
        )
        efficiency_count = category_distribution.get(
            DevianceCategory.EFFICIENCY_GAIN.value, 0
        )
        innovation_count = category_distribution.get(
            DevianceCategory.INNOVATION.value, 0
        )
        unknown_count = category_distribution.get(
            DevianceCategory.UNKNOWN.value, 0
        )

        if violation_count > 0:
            pct = violation_count / total_shadows * 100 if total_shadows > 0 else 0
            recommendations.append(
                f"PROCEDURAL: {violation_count} procedural violations detected "
                f"({pct:.1f}% of shadow events). Consider parliamentarian "
                f"training or procedural review for council members."
            )

        if disruption_count >= 3:
            recommendations.append(
                f"TECHNICAL: {disruption_count} external disruptions recorded. "
                f"Review AV infrastructure and contingency protocols."
            )

        if benign_count > total_shadows * 0.5 and total_shadows > 5:
            recommendations.append(
                "PROCESS: Over half of shadow events are benign flexibility. "
                "Consider formalizing common adaptive behaviors (scheduled breaks, "
                "standard greetings) to reduce perceived deviance."
            )

        if efficiency_count > 0:
            recommendations.append(
                f"POSITIVE: {efficiency_count} efficiency-gain deviations detected. "
                f"Consider adopting these streamlining practices as formal procedure."
            )

        if innovation_count > 0:
            recommendations.append(
                f"POSITIVE: {innovation_count} innovative deliberation approaches "
                f"observed. Evaluate for formal adoption in meeting procedures."
            )

        if unknown_count > total_shadows * 0.3 and total_shadows > 5:
            recommendations.append(
                f"DATA QUALITY: {unknown_count} events could not be classified. "
                f"Consider enabling LLM-based classification or refining "
                f"the rule-based taxonomy for this meeting type."
            )

        if not recommendations:
            recommendations.append(
                "No significant governance concerns identified in shadow events."
            )

        return recommendations

    @staticmethod
    def _empty_summary() -> dict:
        """Return an empty summary structure."""
        return {
            "category_distribution": {},
            "total_shadow_events": 0,
            "total_formal_events": 0,
            "severity_score": 0.0,
            "max_possible_severity": 0.0,
            "severity_percentage": 0.0,
            "top_violations": [],
            "recommendations": ["No shadow events to analyze."],
            "classification_method": {},
        }
