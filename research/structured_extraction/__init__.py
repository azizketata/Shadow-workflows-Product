"""Structured event extraction research module for Meeting Process Twin.

Provides LLM-based structured extraction of (actor, action, object) tuples
from meeting transcripts, with Pydantic schemas for type safety and
evaluation utilities for comparing against keyword-based extraction.

Submodules:
    schema          -- Pydantic models: MeetingEvent, ExtractionResult
    llm_extractor   -- StructuredEventExtractor using GPT-4o-mini structured output
    extraction_eval -- Side-by-side comparison of keyword vs LLM extraction quality
"""
