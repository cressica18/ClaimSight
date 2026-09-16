"""
Business logic services.

Phase 1: Empty. Services added across phases:
- Phase 11: document_intelligence.py — deterministic extraction stub (no PDF parser/OCR wired in)
- Phase 11: pipeline.py             — orchestrates the full analysis pipeline
- Phase 6:  consistency.py          — deterministic rule engine (R1–R9)
- Phase 7:  risk_engine.py          — explainable risk scoring
- Phase 8:  gemini_client.py        — LLM investigation summary
- Phase 5:  cv_service.py           — CV inference (ResNet-50 + demo stub)
- Phase 11: pipeline_locks.py       — concurrency guard for the analysis pipeline

Each service module exposes pure functions or async functions with no
direct FastAPI dependencies, making them independently testable.
"""
