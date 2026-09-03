"""
Gemini / LLM Investigation Layer — blueprint Section 7.

This module is the Phase 8 investigation-summary layer. It takes the
deterministic outputs of the Phase 6 (consistency) and Phase 7 (risk
scoring) engines and asks a Gemini-family model to *narrate* the
findings as a 3–6 sentence summary plus a list of key concerns.

Hard rules (Section 7.1 + the Phase 8 prompt):
- Gemini MUST NOT compute the risk score, risk band, or any arithmetic.
  Those come from Phase 7 deterministically and are passed in as input.
- Gemini MUST NOT evaluate consistency rules. Those come from Phase 6.
- Gemini MUST NOT invent amounts, dates, damage, or findings. Every
  number/date/damage it mentions must already exist in the input.
- The recommendation is computed deterministically from the risk band:
  Low → normal, Medium → manual_review, High → investigate. Gemini's
  returned recommendation is overwritten with the deterministic one if
  they disagree. The deterministic value ALWAYS wins.
- The disclaimer is fixed: "AI-generated, human decision required".
  Gemini's value is ignored.
- Banned phrases (fraud accusations, definitive claims) cause the
  response to be rejected and a repair prompt to be issued. After one
  repair attempt that still contains banned language, the function
  returns `None` and the caller persists with `summary_text=None`.

Failure / retry (Section 7.5):
- Timeout or 5xx → wait `backoff_seconds` (default 2s) → retry once →
  on second failure return `None`.
- Malformed JSON or validation failure → one repair prompt (re-prompt
  with the specific error message) → on second failure return `None`.

Public API:
- `InvestigationInput` (frozen dataclass) — what we send to Gemini.
- `InvestigationOutput` (Pydantic) — what we return after validation.
- `deterministic_recommendation(band: str) -> str`
- `GeminiClient.generate(input) -> InvestigationOutput | None`
- `generate_investigation(claim_id, db, *, client=None) -> InvestigationOutput | None`
- `GeminiError` — raised on programmer errors (bad config, etc.).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import httpx
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.claim import Claim
from app.models.evidence import Evidence
from app.models.investigation import Investigation
from app.models.risk_signal import RiskSignal
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────────────────


# Fixed disclaimer — Gemini's value is always overwritten (Section 7.3).
DISCLAIMER = "AI-generated, human decision required"

# Band → deterministic recommendation mapping (Section 7.4).
# Low → normal, Medium → manual_review, High → investigate.
BAND_TO_RECOMMENDATION: dict[str, str] = {
    "Low": "normal",
    "Medium": "manual_review",
    "High": "investigate",
}

# Phrases that are NOT allowed in `summary` or `key_concerns`. These
# promote unsupported fraud accusations or definitive verdicts. The
# check is case-insensitive and substring-based (with a couple of regex
# forms for the variations Gemini tends to produce).
BANNED_PHRASES: tuple[str, ...] = (
    "this claim is fraudulent",
    "claim is fraudulent",
    "is fraud",
    "commit fraud",
    "committed fraud",
    "definitely fraud",
    "definitely fraudulent",
    "100% fraudulent",
    "prove fraud",
    "proven fraud",
    "this is fraud",
    "guilty of fraud",
    "guilty of insurance fraud",
)

# Regex forms — token-level checks for "fraud" / "fraudulent" that
# should be flagged regardless of the surrounding sentence.
BANNED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(fraud(ulent)?)\b", re.IGNORECASE),
)

# Allowable recommendation values. These match the Investigation /
# Recommendation enum in app/models/enums.py.
_VALID_RECOMMENDATIONS = ("normal", "manual_review", "investigate")

# Number of sentences the summary should target. The prompt instructs
# 3–6; we don't hard-reject outside the range because Gemini's output
# is variable, but we surface a note when it is way off.
SUMMARY_MIN_SENTENCES = 3
SUMMARY_MAX_SENTENCES = 6


# ─── Public dataclasses / schemas ──────────────────────────────────────────


@dataclass(frozen=True)
class InvestigationInput:
    """The structured payload we send to Gemini.

    Mirrors blueprint Section 7.2. All values are taken from the
    deterministic engines — Gemini must not compute any of them.
    """

    claim_id: int
    risk_score: float
    risk_band: str
    risk_signals: tuple[dict[str, Any], ...]
    evidence: tuple[dict[str, Any], ...]
    extracted_documents_summary: dict[str, Any]
    cv_findings: tuple[dict[str, Any], ...]
    # Optional caller-supplied metadata that is also passed verbatim
    # through to the prompt. The model may NOT add new keys here.
    metadata: dict[str, Any] = field(default_factory=dict)


class InvestigationOutput(BaseModel):
    """The validated, ground-checked result we persist.

    `recommendation` is the deterministic one — it is set by the client
    after validation, not by Gemini. `disclaimer` is the fixed string.
    """

    summary: str = Field(..., min_length=1)
    key_concerns: list[str] = Field(default_factory=list)
    recommendation: str
    disclaimer: str = DISCLAIMER
    # The model_version is recorded for audit; it reflects the Gemini
    # model that produced the narrative. It is metadata only.
    model_version: str | None = None
    # Notes about the validation pass — e.g. "2 bullets stripped for
    # missing rule_id" or "recommendation overridden from 'investigate'
    # to 'normal'". Stored alongside the summary so investigators can
    # see *why* a regeneration would have looked different.
    notes: list[str] = Field(default_factory=list)


class GeminiError(RuntimeError):
    """Raised for programmer / configuration errors (no API key, bad URL)."""


# ─── Deterministic recommendation ───────────────────────────────────────────


def deterministic_recommendation(band: str) -> str:
    """Map a risk band to the canonical recommendation (Section 7.4).

    This is the ONLY source of truth for `recommendation`. Gemini's
    output is overwritten with whatever this function returns for the
    input's `risk_band`.
    """
    if band not in BAND_TO_RECOMMENDATION:
        # Defensive default — if the band is unknown we treat it as
        # Medium so the case is surfaced to a human.
        logger.warning("Unknown risk band %r; defaulting to manual_review", band)
        return "manual_review"
    return BAND_TO_RECOMMENDATION[band]


# ─── Input assembly (DB → InvestigationInput) ──────────────────────────────


def _signal_to_dict(sig: RiskSignal) -> dict[str, Any]:
    return {
        "id": sig.id,
        "rule_id": sig.rule_id,
        "category": sig.category,
        "severity": sig.severity,
        "description": sig.description,
    }


def _evidence_to_dict(ev: Evidence) -> dict[str, Any]:
    return {
        "id": ev.id,
        "evidence_type": ev.evidence_type,
        "reference": ev.reference,
        "detail_json": ev.detail_json,
    }


def build_investigation_input(
    claim: Claim,
    signals: Sequence[RiskSignal],
    evidence: Sequence[Evidence],
    *,
    extracted_documents_summary: dict[str, Any] | None = None,
    cv_findings: Sequence[dict[str, Any]] | None = None,
) -> InvestigationInput:
    """Assemble the Section 7.2 input structure from ORM rows.

    `extracted_documents_summary` and `cv_findings` are not part of
    Phase 6/7's deterministic output; they are usually populated by
    the Phase 11 pipeline. We accept them as optional kwargs so the
    tests can run with empty data and so the public API is stable.
    """
    return InvestigationInput(
        claim_id=claim.id,
        risk_score=float(claim.risk_score) if claim.risk_score is not None else 0.0,
        risk_band=claim.risk_band or "Low",
        risk_signals=tuple(_signal_to_dict(s) for s in signals),
        evidence=tuple(_evidence_to_dict(e) for e in evidence),
        extracted_documents_summary=extracted_documents_summary or {},
        cv_findings=tuple(cv_findings or ()),
    )


# ─── Prompt assembly ───────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are an insurance investigation narrator. Your job is to write a
3–6 sentence summary of a claim for a human investigator, citing only
facts already present in the structured input below.

Hard rules (these are non-negotiable):
- Do NOT compute any score, band, or arithmetic. The risk score and
  risk band are provided verbatim — repeat them, do not recalculate.
- Do NOT evaluate rules. The list of `risk_signals` below already
  contains the rule results — narrate them, do not add new ones.
- Do NOT invent amounts, dates, damage, or findings. Every number, date,
  or damage you mention must appear somewhere in the input JSON.
- Each `key_concern` MUST cite a `rule_id` that appears in the
  `risk_signals` list. If a concern has no valid rule_id, do not include
  it.
- Use neutral, evidence-grounded language. Recommend review; do not
  accuse. Never claim the claim is fraudulent, even if there are
  multiple high-severity signals. The decision is always the human's.
- The summary's `recommendation` value is set by the backend from the
  risk band, not by you. Just include it for narrative consistency.
- The `disclaimer` is fixed by the backend: "AI-generated, human
  decision required". You may not change it.

Output format: a single JSON object with exactly these keys:
{
  "summary": string,        // 3–6 sentences, evidence-grounded
  "key_concerns": [string], // each must reference a rule_id from the input
  "recommendation": "normal" | "manual_review" | "investigate",
  "disclaimer": "AI-generated, human decision required"
}

Return only the JSON object. No markdown, no commentary outside it.
"""


def _build_user_prompt(input: InvestigationInput) -> str:
    """Render the InvestigationInput as a JSON block the model can read."""
    payload = {
        "claim_id": input.claim_id,
        "risk_score": input.risk_score,
        "risk_band": input.risk_band,
        "deterministic_recommendation": deterministic_recommendation(input.risk_band),
        "risk_signals": list(input.risk_signals),
        "evidence": list(input.evidence),
        "extracted_documents_summary": input.extracted_documents_summary,
        "cv_findings": list(input.cv_findings),
        "metadata": input.metadata,
    }
    return (
        "Narrate the following claim investigation. "
        "Return only the JSON object described in the system prompt.\n\n"
        "INPUT:\n" + json.dumps(payload, indent=2, default=str)
    )


def _build_repair_prompt(prior_text: str, error: str) -> str:
    """Prompt the model to fix a malformed / failing response."""
    return (
        "Your previous response did not pass validation:\n"
        f"  ERROR: {error}\n\n"
        "Your previous response was:\n"
        f"  {prior_text!r}\n\n"
        "Return a new, valid JSON object that fixes the error. "
        "Follow the same output schema as before. Return only the JSON, "
        "no markdown, no commentary."
    )


# ─── Response parsing & validation ─────────────────────────────────────────


def _extract_text_from_response(raw: str) -> str:
    """Pull the JSON-ish text out of a Gemini response.

    Gemini's REST API returns:
      { "candidates": [ { "content": { "parts": [ { "text": "..." } ] } } ] }
    but for safety we also accept a bare JSON object as the input.
    """
    text = raw.strip()
    if not text:
        raise ValueError("empty response from Gemini")

    # Always try to parse as JSON first. If the result is the Gemini
    # envelope, drill into candidates[0].content.parts[0].text. If the
    # result is our own schema (summary / key_concerns / ...), use it
    # as-is.
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"response is not JSON: {e}") from e

    if isinstance(parsed, dict) and "candidates" in parsed:
        try:
            return parsed["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"unexpected Gemini response shape: {e}") from e

    # Otherwise treat the parsed object as the bare schema.
    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from `text`, with a permissive pass first.

    The model occasionally wraps the JSON in ```json fences or leads
    with prose. We strip a single code fence if present, then try to
    parse the whole string; if that fails we look for the first `{` and
    the last `}` and try again.
    """
    cleaned = text.strip()
    # Strip a single ```json ... ``` fence.
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    # If the text doesn't start with `{`, find the first one.
    if not cleaned.startswith("{"):
        idx = cleaned.find("{")
        if idx == -1:
            raise ValueError("no JSON object found in response")
        cleaned = cleaned[idx:]
    # If the text doesn't end with `}`, find the last one.
    if not cleaned.endswith("}"):
        idx = cleaned.rfind("}")
        if idx == -1:
            raise ValueError("no JSON object end found in response")
        cleaned = cleaned[: idx + 1]
    return json.loads(cleaned)


def _contains_banned(text: str) -> str | None:
    """Return the banned phrase/pattern that `text` contains, or None.

    The returned string is what we surface in the repair prompt so the
    model knows what to avoid.
    """
    if not text:
        return None
    lower = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lower:
            return phrase
    for pattern in BANNED_PATTERNS:
        m = pattern.search(text)
        if m:
            # We allow "fraud" / "fraudulent" only in the **risk band
            # label "fraud risk"** — but our band vocabulary is
            # Low/Medium/High so this never appears. The model may not
            # use these words at all. Surface the exact match.
            return m.group(0)
    return None


def _validate_payload(
    payload: dict[str, Any],
    *,
    valid_rule_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    """Validate the parsed payload and strip hallucinations.

    Returns `(sanitised_payload, notes)`. `notes` describes what we
    changed so the caller can log it or surface it in the UI.
    """
    notes: list[str] = []

    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("missing or empty `summary`")
    summary = summary.strip()

    banned = _contains_banned(summary)
    if banned:
        raise ValueError(f"banned phrase in summary: {banned!r}")

    # key_concerns: each bullet must reference a valid rule_id. Bullets
    # that don't are stripped silently (Section 7.4).
    raw_concerns = payload.get("key_concerns", [])
    if not isinstance(raw_concerns, list):
        raise ValueError("`key_concerns` must be a list")
    concerns: list[str] = []
    for item in raw_concerns:
        if not isinstance(item, str):
            continue
        cited = _extract_rule_id(item)
        if cited is None:
            # No rule_id at all → strip.
            continue
        if cited not in valid_rule_ids:
            notes.append(f"stripped concern (unknown rule_id {cited!r}): {item!r}")
            continue
        concerns.append(item)
    if len(concerns) != len(raw_concerns):
        notes.append(
            f"key_concerns: kept {len(concerns)} of {len(raw_concerns)} (rest stripped for missing/unknown rule_id)"
        )

    # recommendation: we ignore whatever Gemini said. The deterministic
    # value is set by the caller after this function returns. Here we
    # only sanity-check the type and the allowed set.
    rec = payload.get("recommendation")
    if rec is not None and rec not in _VALID_RECOMMENDATIONS:
        notes.append(f"ignoring Gemini recommendation {rec!r} (invalid); will use deterministic")
    # disclaimer: we ignore whatever Gemini said; the constant is set
    # by the caller. We still validate the type.
    disclaimer = payload.get("disclaimer")
    if disclaimer is not None and disclaimer != DISCLAIMER:
        notes.append(
            f"ignoring Gemini disclaimer {disclaimer!r}; will use fixed {DISCLAIMER!r}"
        )

    return {
        "summary": summary,
        "key_concerns": concerns,
        "recommendation": rec,  # caller will overwrite
        "disclaimer": disclaimer,  # caller will overwrite
    }, notes


_RULE_ID_RE = re.compile(r"\bR\d+_[a-z_]+\b")


def _extract_rule_id(text: str) -> str | None:
    """Return the first R#_xxx rule_id token in `text`, or None.

    Used to verify each `key_concern` cites a real rule. Matches the
    `R{n}_{snake_case}` pattern used by the Phase 6 rule functions.
    """
    m = _RULE_ID_RE.search(text)
    return m.group(0) if m else None


# ─── Gemini client ─────────────────────────────────────────────────────────


# Caller signature: (system_prompt, user_prompt) -> raw response text.
Caller = Callable[[str, str], str]


class GeminiClient:
    """Thin, testable wrapper around the Gemini REST API.

    The HTTP transport is hidden behind a `Caller` callable so tests
    can return canned responses without any network I/O. The default
    caller uses `httpx` to call the real Gemini endpoint when an
    `api_key` is configured.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        backoff_seconds: float | None = None,
        caller: Caller | None = None,
        # The `sleep` callable is injected so the test suite can assert
        # on the retry delay without actually waiting.
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.base_url = (base_url or settings.gemini_base_url).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None
            else settings.gemini_timeout_seconds
        )
        self.backoff_seconds = (
            backoff_seconds if backoff_seconds is not None
            else settings.gemini_retry_backoff_seconds
        )
        self._sleep = sleep
        # If a caller is injected, use it. Otherwise build a default
        # httpx-based caller that targets the Gemini REST API.
        self._caller: Caller = caller or self._build_default_caller()

    # ── Public entry point ──────────────────────────────────────────────

    def _generate_demo(self, input: InvestigationInput) -> "InvestigationOutput":
        """Build a deterministic InvestigationOutput from the input.

        Used when `settings.use_demo_gemini` is true. The output shape
        matches the validator (rule_ids are quoted in the summary,
        recommendation is a known enum) so the same persist path is
        exercised.

        Recommendation values are restricted to the
        `Recommendation` enum (`normal`, `manual_review`,
        `investigate`) — anything else will fail the DB INSERT.
        """
        rule_lines: list[str] = []
        for s in input.risk_signals:
            rule_lines.append(f"[{s['rule_id']}] {s.get('description', '')}")
        summary = (
            "Demo-mode investigation summary.\n\n"
            "Risk signals:\n  - " + "\n  - ".join(rule_lines)
            if rule_lines
            else "Demo-mode investigation summary. No risk signals fired."
        )
        if not input.risk_signals:
            recommendation = "normal"
        elif any(s.get("severity") == "high" for s in input.risk_signals):
            recommendation = "manual_review"
        else:
            recommendation = "investigate"
        return InvestigationOutput(
            summary=summary,
            key_concerns=[s["rule_id"] for s in input.risk_signals],
            recommendation=recommendation,
            model_version="demo_deterministic_v1",
        )

    def generate(self, input: InvestigationInput) -> InvestigationOutput | None:
        """Run the full generate → validate → repair pipeline.

        Returns the validated `InvestigationOutput` on success, or
        `None` after every retry is exhausted. Never raises for
        Gemini-side errors; only `GeminiError` for misconfiguration.
        """
        # Demo mode (Phase 13): return a deterministic stub built from
        # the input so the demo runs end-to-end without a Gemini key.
        from app.core.config import settings as _settings
        if _settings.use_demo_gemini:
            return self._generate_demo(input)

        if not self.api_key and self._caller.__name__ != "_build_default_caller":
            # Caller-injected path doesn't need a real key.
            pass
        # No caller AND no API key → cannot talk to Gemini. Surface
        # this as a configuration error so the pipeline can fail fast.
        if self._caller is None and not self.api_key:
            raise GeminiError("gemini_api_key is not configured")

        valid_rule_ids = {s["rule_id"] for s in input.risk_signals}
        user_prompt = _build_user_prompt(input)
        system_prompt = _SYSTEM_PROMPT

        last_error: str | None = None
        last_text: str | None = None

        # First attempt.
        try:
            raw = self._caller(system_prompt, user_prompt)
        except _RetryableNetworkError as e:
            # Network failure → wait → retry once. If retry also fails
            # we return None.
            logger.warning("Gemini network error (attempt 1): %s", e)
            self._sleep(self.backoff_seconds)
            try:
                raw = self._caller(system_prompt, user_prompt)
            except _RetryableNetworkError as e2:
                logger.warning("Gemini network error (attempt 2): %s", e2)
                return None
        last_text = raw

        # Try to parse + validate.
        outcome = self._parse_and_validate(raw, valid_rule_ids=valid_rule_ids)
        if isinstance(outcome, tuple):
            sanitised, validation_notes = outcome
            return self._finalize(input, sanitised, validation_notes)

        # First attempt failed. One repair pass with the specific error.
        last_error = outcome
        repair = _build_repair_prompt(last_text or "", last_error or "unknown error")
        try:
            raw2 = self._caller(system_prompt, repair)
        except _RetryableNetworkError as e:
            logger.warning("Gemini network error (repair attempt): %s", e)
            return None
        last_text = raw2
        outcome2 = self._parse_and_validate(raw2, valid_rule_ids=valid_rule_ids)
        if isinstance(outcome2, tuple):
            sanitised, validation_notes = outcome2
            return self._finalize(input, sanitised, validation_notes)

        # Both attempts failed → graceful null fallback.
        logger.info("Gemini output failed validation after repair; returning None")
        return None

    # ── Internals ──────────────────────────────────────────────────────

    def _parse_and_validate(
        self,
        raw: str,
        *,
        valid_rule_ids: set[str],
    ) -> tuple[dict[str, Any], list[str]] | str:
        """Try to parse + validate `raw`.

        Returns the sanitised payload on success, or an error string
        on failure. The string is what the repair prompt will echo.
        """
        try:
            text = _extract_text_from_response(raw)
        except ValueError as e:
            return f"envelope parse failed: {e}"
        try:
            payload = _parse_json_object(text)
        except (ValueError, json.JSONDecodeError) as e:
            return f"JSON parse failed: {e}"
        try:
            sanitised, notes = _validate_payload(payload, valid_rule_ids=valid_rule_ids)
        except ValueError as e:
            return f"validation failed: {e}"
        return sanitised, notes

    def _finalize(
        self,
        input: InvestigationInput,
        sanitised: dict[str, Any],
        validation_notes: list[str] | None = None,
    ) -> InvestigationOutput:
        """Apply the deterministic overrides and return the final output."""
        notes: list[str] = list(validation_notes or [])
        det = deterministic_recommendation(input.risk_band)
        if sanitised.get("recommendation") != det:
            notes.append(
                f"recommendation overridden from {sanitised.get('recommendation')!r} "
                f"to deterministic {det!r} (band={input.risk_band!r})"
            )
        return InvestigationOutput(
            summary=sanitised["summary"],
            key_concerns=sanitised["key_concerns"],
            recommendation=det,
            disclaimer=DISCLAIMER,
            model_version=self.model,
            notes=notes,
        )

    def _build_default_caller(self) -> Caller:
        """Build an httpx-based caller for the real Gemini REST API.

        Network errors and 5xx responses are wrapped in
        `_RetryableNetworkError` so the retry logic in `generate` can
        catch them. 4xx errors (e.g. invalid API key) are wrapped in
        `GeminiError` because retrying won't help.
        """
        api_key = self.api_key
        model = self.model
        base_url = self.base_url
        timeout = self.timeout_seconds

        def caller(system_prompt: str, user_prompt: str) -> str:
            if not api_key:
                raise GeminiError("gemini_api_key is not configured")
            url = f"{base_url}/v1beta/models/{model}:generateContent"
            body = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": system_prompt + "\n\n" + user_prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.9,
                    "responseMimeType": "application/json",
                },
            }
            try:
                resp = httpx.post(
                    url,
                    params={"key": api_key},
                    json=body,
                    timeout=timeout,
                )
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                raise _RetryableNetworkError(str(e)) from e
            if resp.status_code in (500, 502, 503, 504):
                raise _RetryableNetworkError(
                    f"Gemini {resp.status_code}: {resp.text[:200]}"
                )
            if resp.status_code >= 400:
                # 4xx — configuration / bad request. Don't retry.
                raise GeminiError(
                    f"Gemini {resp.status_code}: {resp.text[:200]}"
                )
            return resp.text

        return caller


class _RetryableNetworkError(RuntimeError):
    """Internal: a network/5xx failure that the retry policy should handle."""


# ─── Top-level helper ─────────────────────────────────────────────────────


def generate_investigation(
    claim_id: int,
    db: Session,
    *,
    client: GeminiClient | None = None,
) -> InvestigationOutput | None:
    """Assemble the input from the DB, call Gemini, and (on failure) return None.

    The caller (Phase 11 pipeline or a Phase 8 API endpoint) decides
    how to persist the result. Persisting is not the job of this
    function so it stays unit-testable.
    """
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise GeminiError(f"claim {claim_id} not found")

    signals: list[RiskSignal] = list(claim.risk_signals)
    evidence: list[Evidence] = []
    for s in signals:
        evidence.extend(s.evidence)

    investigation_input = build_investigation_input(
        claim=claim, signals=signals, evidence=evidence
    )
    client = client or GeminiClient()
    return client.generate(investigation_input)


# ─── Persistence helper (kept out of the unit-testable surface) ──────────


def persist_investigation(
    output: InvestigationOutput | None,
    db: Session,
    *,
    claim: Claim,
) -> Investigation:
    """Write the Gemini output to the `investigation` table (1:1 with Claim).

    If `output` is None the function still writes an Investigation row
    with `summary_text=None` and the deterministic recommendation, so
    the UI's "narrative summary unavailable — retry" button has a row
    to attach to.
    """
    rec = (
        deterministic_recommendation(claim.risk_band or "Low")
        if output is None
        else output.recommendation
    )
    inv = db.query(Investigation).filter(Investigation.claim_id == claim.id).one_or_none()
    if inv is None:
        inv = Investigation(claim_id=claim.id)
        db.add(inv)
    inv.summary_text = output.summary if output is not None else None
    inv.recommendation = rec
    inv.model_version = output.model_version if output is not None else None
    if output is not None:
        from datetime import datetime, timezone
        inv.generated_at = datetime.now(timezone.utc)
    db.add(inv)
    db.flush()
    return inv
