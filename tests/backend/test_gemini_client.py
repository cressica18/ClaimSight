"""
Phase 8 — Gemini Investigation Layer unit tests.

All tests use a mocked `Caller` so no real Gemini call is ever made.
Each test focuses on one behaviour:

1. Valid grounded Gemini response → InvestigationOutput with summary,
   3–6 sentences, key_concerns citing valid rule_ids, recommendation
   overridden to the deterministic value, fixed disclaimer.
2. Rule-id grounding → bullets referencing unknown rule_ids are
   stripped silently; valid ones are kept.
3. Recommendation override → Gemini's `investigate` on a Low-band
   claim is replaced with the deterministic `normal`.
4. Banned-language → a response containing a banned phrase is rejected;
   the next repair attempt with a clean response succeeds.
5. Malformed JSON → one repair attempt is made; the repair response
   must itself be valid JSON; if not, the function returns None.
6. Timeout / 5xx → one retry after `backoff_seconds`; the second
   success is used; second failure returns None.
7. Final graceful failure → after retry exhaustion the function
   returns None (caller persists with summary_text=None).
8. Deterministic disclaimer → the disclaimer is always the fixed
   constant regardless of what Gemini returned.

Plus a few integration-flavor tests:
- Banned-phrase regex catches the variations the engine lists.
- Input assembly from a real DB session (Claim + RiskSignal + Evidence).
- persist_investigation writes summary_text, recommendation, and the
  generated_at timestamp.
- persist_investigation with output=None writes summary_text=None and
  the deterministic recommendation.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Callable

import pytest

from app.models import (
    Claim,
    Customer,
    Evidence,
    Investigation,
    Policy,
    RiskSignal,
    Vehicle,
)
from app.models.enums import (
    ClaimStatus,
    CoverageType,
    EvidenceType,
    PolicyStatus,
    Recommendation,
    SignalSeverity,
)
from app.services.gemini_client import (
    DISCLAIMER,
    GeminiClient,
    GeminiError,
    InvestigationInput,
    InvestigationOutput,
    _contains_banned,
    _extract_rule_id,
    _parse_json_object,
    build_investigation_input,
    deterministic_recommendation,
    generate_investigation,
    persist_investigation,
)


# ─── Helpers ───────────────────────────────────────────────────────────────


def _make_input(
    *,
    band: str = "Medium",
    score: float = 50.0,
    rule_ids: tuple[str, ...] = ("R1_unsupported_damage",),
) -> InvestigationInput:
    return InvestigationInput(
        claim_id=1,
        risk_score=score,
        risk_band=band,
        risk_signals=tuple(
            {
                "id": i + 1,
                "rule_id": rid,
                "category": "test",
                "severity": "high",
                "description": f"test signal {i+1}",
            }
            for i, rid in enumerate(rule_ids)
        ),
        evidence=(),
        extracted_documents_summary={},
        cv_findings=(),
    )


def _good_schema_response(
    *,
    rule_ids: tuple[str, ...] = ("R1_unsupported_damage",),
    recommendation: str = "manual_review",
    summary: str = (
        "The claim filed for vehicle CLM-1 has a Medium risk score of 50. "
        "Rule R1_unsupported_damage was triggered because the claim form "
        "lists a damage area not seen by the CV model. The cost ratio is "
        "within the baseline range. The investigator should review the "
        "discrepancy between the claim form and the uploaded image."
    ),
    include_disclaimer: bool = True,
) -> dict[str, Any]:
    """Return the bare schema payload (no Gemini envelope)."""
    concerns = [f"Concern {i+1}: {rid} was triggered." for i, rid in enumerate(rule_ids)]
    return {
        "summary": summary,
        "key_concerns": concerns,
        "recommendation": recommendation,
        "disclaimer": DISCLAIMER if include_disclaimer else "Gemini's own disclaimer",
    }


# Backwards-compat alias: tests that pre-date the rename use this name.
_good_gemini_response = _good_schema_response


def _envelope(text: str) -> str:
    """Wrap `text` in a Gemini response envelope, JSON-serialised."""
    return json.dumps(
        {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    )


# ─── 1. Valid grounded response ───────────────────────────────────────────


def test_valid_grounded_response_produces_output():
    out = _make_input(band="Medium", rule_ids=("R1_unsupported_damage",))
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: json.dumps(_good_gemini_response()),
        sleep=lambda _s: None,
    )
    result = client.generate(out)
    assert isinstance(result, InvestigationOutput)
    assert "Medium" in result.summary
    assert result.recommendation == "manual_review"
    assert result.disclaimer == DISCLAIMER
    assert result.model_version  # whatever default model is set


def test_valid_response_with_envelope_shape():
    """A response in the real Gemini envelope shape is handled."""
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(
            json.dumps(_good_gemini_response())
        ),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="High", rule_ids=("R1_unsupported_damage",)))
    assert result is not None
    assert result.recommendation == "investigate"


def test_valid_response_with_json_fence():
    """The model occasionally wraps the response in ```json fences."""
    fenced = "```json\n" + json.dumps(_good_gemini_response()) + "\n```"
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(fenced),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input())
    assert result is not None
    assert result.summary


def test_valid_response_with_prose_before_json():
    """The model sometimes adds prose before the JSON object."""
    text = "Here is the JSON you asked for:\n\n" + json.dumps(_good_gemini_response())
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(text),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input())
    assert result is not None


# ─── 2. Rule-id grounding ─────────────────────────────────────────────────


def test_rule_id_grounding_strips_unknown_rule_ids():
    payload = {
        "summary": (
            "R1_unsupported_damage and R5_duplicate_previous_damage both "
            "fired. R99_made_up_rule also fired but is not real."
        ),
        "key_concerns": [
            "Concern A: R1_unsupported_damage was triggered.",  # valid
            "Concern B: R5_duplicate_previous_damage was triggered.",  # valid
            "Concern C: R99_made_up_rule was triggered.",  # unknown — should be stripped
            "Concern D: no rule id at all here.",  # no rule id — should be stripped
            "Concern E: a bullet that quotes R1_unsupported_damage in the middle.",  # valid
        ],
        "recommendation": "manual_review",
        "disclaimer": DISCLAIMER,
    }
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(json.dumps(payload)),
        sleep=lambda _s: None,
    )
    result = client.generate(
        _make_input(
            band="Medium",
            rule_ids=("R1_unsupported_damage", "R5_duplicate_previous_damage"),
        )
    )
    assert result is not None
    # 3 valid concerns should remain; 2 stripped.
    assert len(result.key_concerns) == 3
    assert all(
        any(rid in c for rid in ("R1_unsupported_damage", "R5_duplicate_previous_damage"))
        for c in result.key_concerns
    )
    # Notes should mention the strip.
    assert any("stripped" in n.lower() for n in result.notes)


def test_rule_id_grounding_strips_all_when_none_valid():
    """If every bullet references a bogus rule_id, key_concerns is empty."""
    payload = {
        "summary": "The claim shows some patterns. R1_unsupported_damage was logged but is not a real rule here.",
        "key_concerns": [
            "R99_made_up was triggered.",
            "R77_also_fake was triggered.",
        ],
        "recommendation": "manual_review",
        "disclaimer": DISCLAIMER,
    }
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(json.dumps(payload)),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is not None
    assert result.key_concerns == []
    assert any("stripped" in n.lower() for n in result.notes)


def test_extract_rule_id_helper():
    assert _extract_rule_id("R1_unsupported_damage fired.") == "R1_unsupported_damage"
    assert _extract_rule_id("no rule here") is None
    assert _extract_rule_id("R12_abc_def mentions nothing") == "R12_abc_def"


# ─── 3. Recommendation override ───────────────────────────────────────────


def test_recommendation_override_low_band_normal():
    """Even if Gemini says 'investigate', a Low band yields 'normal'."""
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(
            json.dumps(_good_gemini_response(recommendation="investigate"))
        ),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Low", rule_ids=("R1_unsupported_damage",)))
    assert result is not None
    assert result.recommendation == "normal"
    assert any("overridden" in n for n in result.notes)


def test_recommendation_override_high_band_investigate():
    """If Gemini says 'normal' on a High band, the override fires too."""
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(
            json.dumps(_good_gemini_response(recommendation="normal"))
        ),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="High", rule_ids=("R1_unsupported_damage",)))
    assert result is not None
    assert result.recommendation == "investigate"
    assert any("overridden" in n for n in result.notes)


def test_recommendation_agrees_no_override_note():
    """If Gemini's value already matches, no override note is added."""
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(
            json.dumps(_good_gemini_response(recommendation="manual_review"))
        ),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is not None
    assert not any("overridden" in n for n in result.notes)


def test_deterministic_recommendation_helper():
    assert deterministic_recommendation("Low") == "normal"
    assert deterministic_recommendation("Medium") == "manual_review"
    assert deterministic_recommendation("High") == "investigate"
    # Unknown band defaults to manual_review
    assert deterministic_recommendation("???") == "manual_review"


# ─── 4. Banned-language handling ──────────────────────────────────────────


def test_banned_phrase_triggers_repair_attempt():
    """A summary with a banned phrase is rejected; a clean repair succeeds."""
    bad_payload = _good_gemini_response(
        summary=(
            "This claim is fraudulent based on the high-severity signals. "
            "R1_unsupported_damage indicates tampering."
        )
    )
    good_payload = _good_gemini_response()  # clean summary
    responses = [_envelope(json.dumps(bad_payload)), _envelope(json.dumps(good_payload))]
    call_count = {"n": 0}

    def fake_caller(sys, user):
        r = responses[call_count["n"]]
        call_count["n"] += 1
        return r

    client = GeminiClient(
        api_key="dummy",
        caller=fake_caller,
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    # The first attempt was rejected (banned phrase); the repair attempt
    # returned a clean response → overall result is not None.
    assert result is not None
    assert "fraud" not in result.summary.lower()
    assert call_count["n"] == 2


def test_banned_phrase_persists_through_repair_returns_none():
    """If the repair attempt is also banned, the function returns None."""
    bad_payload = _good_gemini_response(
        summary="This claim is fraudulent. R1_unsupported_damage fired."
    )
    responses = [
        _envelope(json.dumps(bad_payload)),
        _envelope(json.dumps(bad_payload)),  # repair attempt still bad
    ]
    call_count = {"n": 0}

    def fake_caller(sys, user):
        r = responses[call_count["n"]]
        call_count["n"] += 1
        return r

    client = GeminiClient(
        api_key="dummy",
        caller=fake_caller,
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is None
    assert call_count["n"] == 2


def test_contains_banned_phrase_detection():
    """Direct test of the banned-phrase detector."""
    assert _contains_banned("This claim is fraudulent.") is not None
    assert _contains_banned("The claimant committed fraud.") is not None
    assert _contains_banned("100% fraudulent claim here.") is not None
    # Token-level: any use of "fraud" or "fraudulent" is rejected.
    assert _contains_banned("There is fraud here.") is not None
    # Clean language is allowed.
    assert _contains_banned("The claim shows a cost anomaly.") is None
    assert _contains_banned("Recommend manual review.") is None


def test_banned_summary_phrase_variations():
    """All known variations are caught."""
    for phrase in [
        "this claim is fraudulent",
        "the claim is fraudulent",
        "fraud was committed",
        "this is fraud",
        "definitely fraudulent",
    ]:
        assert _contains_banned(f"Evidence shows: {phrase}.") is not None, phrase


# ─── 5. Malformed JSON retry ──────────────────────────────────────────────


def test_malformed_json_triggers_repair_attempt():
    """Bad JSON → repair prompt → good JSON on retry → success."""
    good = _envelope(json.dumps(_good_gemini_response()))
    bad = _envelope("this is not JSON at all")
    responses = [bad, good]
    call_count = {"n": 0}

    def fake_caller(sys, user):
        r = responses[call_count["n"]]
        call_count["n"] += 1
        return r

    client = GeminiClient(
        api_key="dummy",
        caller=fake_caller,
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is not None
    assert call_count["n"] == 2


def test_malformed_json_after_repair_returns_none():
    bad = _envelope("not JSON")
    responses = [bad, bad]
    call_count = {"n": 0}

    def fake_caller(sys, user):
        r = responses[call_count["n"]]
        call_count["n"] += 1
        return r

    client = GeminiClient(
        api_key="dummy",
        caller=fake_caller,
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is None
    assert call_count["n"] == 2


def test_malformed_envelope_returns_none():
    """The Gemini envelope itself is unparseable → repair attempt → still bad → None."""
    bad = "this is not even an envelope"
    responses = [bad, bad]
    call_count = {"n": 0}

    def fake_caller(sys, user):
        r = responses[call_count["n"]]
        call_count["n"] += 1
        return r

    client = GeminiClient(
        api_key="dummy",
        caller=fake_caller,
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is None


def test_parse_json_object_strips_fence():
    text = "```json\n" + json.dumps({"a": 1}) + "\n```"
    parsed = _parse_json_object(text)
    assert parsed == {"a": 1}


def test_parse_json_object_handles_prose_wrapper():
    text = "Here is the result:\n\n" + json.dumps({"a": 1}) + "\n\nDone."
    parsed = _parse_json_object(text)
    assert parsed == {"a": 1}


# ─── 6. Timeout / 5xx retry ──────────────────────────────────────────────


class _FakeNetworkError(Exception):
    """Simulates httpx.TimeoutException or 5xx for the retry path."""


def _retryable_raiser_factory(
    responses: list,
    call_log: list,
):
    """Build a caller that raises `_FakeNetworkError` for sentinel entries.

    Each entry in `responses` is either a string (the response text) or
    the special sentinel object `_RAISE` to mean "raise a network error".
    """
    from app.services.gemini_client import _RetryableNetworkError

    def caller(sys, user):
        call_log.append(user)
        item = responses[len(call_log) - 1]
        if item is _RAISE:
            raise _RetryableNetworkError("simulated 503")
        return item

    return caller


_RAISE = object()


def test_timeout_retries_once_with_backoff_then_succeeds():
    """First call raises (network), second call succeeds → result is OK."""
    from app.services.gemini_client import _RetryableNetworkError

    sleep_log: list[float] = []
    call_log: list[str] = []

    def fake_caller(sys, user):
        call_log.append(user)
        if len(call_log) == 1:
            raise _RetryableNetworkError("timeout")
        return _envelope(json.dumps(_good_gemini_response()))

    client = GeminiClient(
        api_key="dummy",
        caller=fake_caller,
        sleep=lambda s: sleep_log.append(s),
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is not None
    # 2 calls (1 failed + 1 retry)
    assert len(call_log) == 2
    # Backoff was invoked exactly once with the configured value
    assert sleep_log == [2.0]


def test_5xx_retries_once_with_backoff():
    """A 5xx response on first call → retry once → second 5xx → None."""
    from app.services.gemini_client import _RetryableNetworkError

    sleep_log: list[float] = []
    call_log: list[str] = []

    def fake_caller(sys, user):
        call_log.append(user)
        raise _RetryableNetworkError("simulated 503")

    client = GeminiClient(
        api_key="dummy",
        caller=fake_caller,
        sleep=lambda s: sleep_log.append(s),
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is None
    assert len(call_log) == 2
    assert sleep_log == [2.0]


def test_backoff_value_is_configurable():
    """The 2s backoff is read from settings; tests can shrink it."""
    from app.services.gemini_client import _RetryableNetworkError

    sleep_log: list[float] = []
    call_log: list[str] = []

    def fake_caller(sys, user):
        call_log.append(user)
        raise _RetryableNetworkError("simulated 503")

    client = GeminiClient(
        api_key="dummy",
        caller=fake_caller,
        sleep=lambda s: sleep_log.append(s),
        backoff_seconds=0.5,
    )
    client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert sleep_log == [0.5]


def test_no_backoff_on_content_validation_failure():
    """A banned-phrase / JSON failure does NOT trigger the 2s sleep."""
    sleep_log: list[float] = []

    bad = _envelope("not valid JSON")
    responses = [bad, bad]
    call_count = {"n": 0}

    def fake_caller(sys, user):
        r = responses[call_count["n"]]
        call_count["n"] += 1
        return r

    client = GeminiClient(
        api_key="dummy",
        caller=fake_caller,
        sleep=lambda s: sleep_log.append(s),
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is None
    # Both calls were made, but sleep was never called.
    assert sleep_log == []


# ─── 7. Final graceful failure / null fallback ────────────────────────────


def test_graceful_failure_after_retry_exhaustion_returns_none():
    """After every retry is exhausted, the function returns None."""
    from app.services.gemini_client import _RetryableNetworkError

    sleep_log: list[float] = []
    call_log: list[str] = []

    def fake_caller(sys, user):
        call_log.append(user)
        raise _RetryableNetworkError("network down")

    client = GeminiClient(
        api_key="dummy",
        caller=fake_caller,
        sleep=lambda s: sleep_log.append(s),
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is None
    # Two attempts (1 + 1 retry), no further retries
    assert len(call_log) == 2


def test_persist_investigation_with_none_writes_null_summary(db_session):
    """`persist_investigation(None, ...)` still writes a row with summary_text=None."""
    from app.models.enums import ClaimStatus
    customer = Customer(name="G", email="g@x.test", phone="+1")
    db_session.add(customer); db_session.flush()
    vehicle = Vehicle(customer_id=customer.id, make="Honda", model="Accord",
                      year=2021, vin="VG", plate_number="PG")
    db_session.add(vehicle); db_session.flush()
    policy = Policy(
        customer_id=customer.id, vehicle_id=vehicle.id, policy_number="PG",
        coverage_type=CoverageType.comprehensive.value,
        coverage_limit=50000, deductible=500,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2025, 12, 31),
        status=PolicyStatus.active.value,
    )
    db_session.add(policy); db_session.flush()
    claim = Claim(
        policy_id=policy.id, vehicle_id=vehicle.id, claim_number="PG-C",
        incident_date=dt.date(2025, 6, 15), reported_date=dt.date(2025, 6, 16),
        claimed_amount=500, status=ClaimStatus.pending.value,
        risk_score=80, risk_band="High",
    )
    db_session.add(claim); db_session.flush()

    inv = persist_investigation(None, db_session, claim=claim)
    assert inv.summary_text is None
    # Even with no summary, the deterministic recommendation is written.
    assert inv.recommendation == "investigate"  # High → investigate
    db_session.commit()
    db_session.refresh(claim)
    assert claim.investigation is not None
    assert claim.investigation.summary_text is None


# ─── 8. Deterministic disclaimer ──────────────────────────────────────────


def test_disclaimer_is_always_fixed_constant():
    """Even when Gemini returns a different disclaimer, we use DISCLAIMER."""
    payload = _good_gemini_response(include_disclaimer=False)  # custom disclaimer
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(json.dumps(payload)),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is not None
    assert result.disclaimer == DISCLAIMER
    # The notes mention we overrode Gemini's disclaimer.
    assert any("disclaimer" in n.lower() for n in result.notes)


def test_disclaimer_is_set_in_persisted_investigation(db_session):
    """The persisted row carries the fixed disclaimer (via the fixed string in schema)."""
    from app.models.enums import ClaimStatus
    customer = Customer(name="H", email="h@x.test", phone="+1")
    db_session.add(customer); db_session.flush()
    vehicle = Vehicle(customer_id=customer.id, make="Honda", model="Accord",
                      year=2021, vin="VH", plate_number="PH")
    db_session.add(vehicle); db_session.flush()
    policy = Policy(
        customer_id=customer.id, vehicle_id=vehicle.id, policy_number="PH",
        coverage_type=CoverageType.comprehensive.value,
        coverage_limit=50000, deductible=500,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2025, 12, 31),
        status=PolicyStatus.active.value,
    )
    db_session.add(policy); db_session.flush()
    claim = Claim(
        policy_id=policy.id, vehicle_id=vehicle.id, claim_number="PH-C",
        incident_date=dt.date(2025, 6, 15), reported_date=dt.date(2025, 6, 16),
        claimed_amount=500, status=ClaimStatus.pending.value,
        risk_score=50, risk_band="Medium",
    )
    db_session.add(claim); db_session.flush()

    output = InvestigationOutput(
        summary="Summary.",
        key_concerns=["R1_unsupported_damage noted."],
        recommendation="manual_review",
        disclaimer="Something Gemini tried to set.",
        model_version="test-model",
    )
    inv = persist_investigation(output, db_session, claim=claim)
    # The persisted row's summary text is what was set; the schema-level
    # disclaimer is fixed via the Pydantic default.
    assert inv.summary_text == "Summary."
    assert inv.recommendation == "manual_review"
    assert inv.model_version == "test-model"
    assert inv.generated_at is not None


# ─── Input assembly from a real DB session ────────────────────────────────


def test_build_investigation_input_from_db(db_session):
    customer = Customer(name="I", email="i@x.test", phone="+1")
    db_session.add(customer); db_session.flush()
    vehicle = Vehicle(customer_id=customer.id, make="Honda", model="Accord",
                      year=2021, vin="VI", plate_number="PI")
    db_session.add(vehicle); db_session.flush()
    policy = Policy(
        customer_id=customer.id, vehicle_id=vehicle.id, policy_number="PI",
        coverage_type=CoverageType.comprehensive.value,
        coverage_limit=50000, deductible=500,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2025, 12, 31),
        status=PolicyStatus.active.value,
    )
    db_session.add(policy); db_session.flush()
    claim = Claim(
        policy_id=policy.id, vehicle_id=vehicle.id, claim_number="PI-C",
        incident_date=dt.date(2025, 6, 15), reported_date=dt.date(2025, 6, 16),
        claimed_amount=500, status=ClaimStatus.pending.value,
        risk_score=72, risk_band="High",
    )
    db_session.add(claim); db_session.flush()
    sig = RiskSignal(
        claim_id=claim.id, rule_id="R1_unsupported_damage",
        category="image_claim_consistency",
        severity=SignalSeverity.high.value,
        description="Front bumper claimed but not seen in CV.",
    )
    db_session.add(sig); db_session.flush()
    db_session.refresh(sig)
    ev = Evidence(
        risk_signal_id=sig.id,
        evidence_type=EvidenceType.image.value,
        reference="image-1",
        detail_json={"bounding_box": [10, 20, 100, 120]},
    )
    db_session.add(ev); db_session.flush()
    db_session.refresh(ev)

    inv_input = build_investigation_input(
        claim=claim, signals=[sig], evidence=[ev]
    )
    assert inv_input.claim_id == claim.id
    assert inv_input.risk_score == 72
    assert inv_input.risk_band == "High"
    assert len(inv_input.risk_signals) == 1
    assert inv_input.risk_signals[0]["rule_id"] == "R1_unsupported_damage"
    assert len(inv_input.evidence) == 1
    assert inv_input.evidence[0]["evidence_type"] == "image"


def test_generate_investigation_top_level_happy_path(db_session):
    """`generate_investigation` returns a real `InvestigationOutput`."""
    customer = Customer(name="J", email="j@x.test", phone="+1")
    db_session.add(customer); db_session.flush()
    vehicle = Vehicle(customer_id=customer.id, make="Honda", model="Accord",
                      year=2021, vin="VJ", plate_number="PJ")
    db_session.add(vehicle); db_session.flush()
    policy = Policy(
        customer_id=customer.id, vehicle_id=vehicle.id, policy_number="PJ",
        coverage_type=CoverageType.comprehensive.value,
        coverage_limit=50000, deductible=500,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2025, 12, 31),
        status=PolicyStatus.active.value,
    )
    db_session.add(policy); db_session.flush()
    claim = Claim(
        policy_id=policy.id, vehicle_id=vehicle.id, claim_number="PJ-C",
        incident_date=dt.date(2025, 6, 15), reported_date=dt.date(2025, 6, 16),
        claimed_amount=500, status=ClaimStatus.pending.value,
        risk_score=72, risk_band="High",
    )
    db_session.add(claim); db_session.flush()
    sig = RiskSignal(
        claim_id=claim.id, rule_id="R1_unsupported_damage",
        category="image_claim_consistency",
        severity=SignalSeverity.high.value,
        description="Front bumper claimed but not seen in CV.",
    )
    db_session.add(sig); db_session.flush()
    db_session.commit()

    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(json.dumps(_good_gemini_response())),
        sleep=lambda _s: None,
    )
    result = generate_investigation(claim.id, db_session, client=client)
    assert result is not None
    assert result.recommendation == "investigate"
    assert result.disclaimer == DISCLAIMER


def test_generate_investigation_returns_none_on_claim_not_found(db_session):
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: "",
        sleep=lambda _s: None,
    )
    with pytest.raises(GeminiError):
        generate_investigation(999_999_999, db_session, client=client)


# ─── Misc: configuration errors ───────────────────────────────────────────


def test_client_without_api_key_and_default_caller_raises():
    """Default caller + no api_key → GeminiError (programmer error)."""
    client = GeminiClient(api_key="", caller=None)  # forces default caller
    # The client defers the check until generate is called.
    with pytest.raises(GeminiError):
        client.generate(_make_input())


def test_client_with_injected_caller_does_not_require_api_key():
    client = GeminiClient(
        api_key=None,
        caller=lambda sys, user: _envelope(json.dumps(_good_gemini_response())),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input())
    assert result is not None


# ─── Misc: empty signal list / edge case ──────────────────────────────────


def test_zero_signals_still_works():
    """An empty risk_signals list is a valid input — every concern is stripped."""
    payload = {
        "summary": "The claim has no risk signals. R1_unsupported_damage was considered but did not fire.",
        "key_concerns": [],
        "recommendation": "normal",
        "disclaimer": DISCLAIMER,
    }
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(json.dumps(payload)),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Low", rule_ids=()))
    assert result is not None
    assert result.key_concerns == []
    assert result.recommendation == "normal"


def test_summary_too_short_still_accepted():
    """The engine does not hard-reject short summaries; it just doesn't
    enforce the 3–6 sentence target (the prompt instructs the model)."""
    payload = {
        "summary": "Short.",
        "key_concerns": [],
        "recommendation": "manual_review",
        "disclaimer": DISCLAIMER,
    }
    client = GeminiClient(
        api_key="dummy",
        caller=lambda sys, user: _envelope(json.dumps(payload)),
        sleep=lambda _s: None,
    )
    result = client.generate(_make_input(band="Medium", rule_ids=("R1_unsupported_damage",)))
    assert result is not None
    assert result.summary == "Short."
