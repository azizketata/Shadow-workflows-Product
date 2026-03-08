"""Process deviance classification for Meeting Process Twin.

Classifies shadow workflow events into a taxonomy of deviance types
(benign flexibility, procedural violations, efficiency gains, etc.)
and generates structured reports for thesis-quality analysis.

Modules:
    deviance_taxonomy   - Enum taxonomy and rule-based keyword classification
    deviance_classifier - Hybrid rules + LLM classifier for shadow events
    deviance_report     - Markdown report generation for deviance analysis
"""
