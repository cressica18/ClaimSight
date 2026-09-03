"""
Phase 13 — cv_service regression tests.

The `run_cv_on_image` function takes an optional `predictor` argument
that callers (the pipeline, the demo data script, the test suite) use
to inject a fake / trained predictor and avoid the heavyweight
`ml.inference.predictor` import. A regression in Phase 13 surfaced:
the unconditional `from ml.inference.predictor import …` at the top
of the function meant the injection point was effectively dead — any
runtime that did not have the `ml` package on sys.path (the demo
script, a slim Docker image) would crash with `ModuleNotFoundError`
before the injected predictor was consulted.

These tests pin the contract: when a predictor is injected, the
function must NOT import the real `ml` package, and the result rows
must reflect the injected predictor's output.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from app.models.customer import Customer
from app.models.policy import Policy
from app.models.vehicle import Vehicle
from app.models.claim import Claim
from app.models.damage import Damage

from app.services import cv_service


class _InjectedPredictor:
    """A stand-in for `ml.inference.predictor.VehicleDamagePredictor`.

    Builds duck-typed objects that match the CVPrediction contract
    (`.damage_types[*].label/.confidence`, `.severity.label/.confidence`,
    `.low_confidence`, `.model_version`, `.source_image`, `.timestamp`,
    `.error`) without importing from `ml` at all — otherwise the test
    cannot simulate an environment where `ml` is unavailable.
    """

    def __init__(self, damage_type: str, severity: str):
        self.damage_type = damage_type
        self.severity = severity
        self.calls: list[Path] = []

    def predict_from_path(self, image_path):  # noqa: D401
        self.calls.append(Path(image_path))

        class _Pred:
            def __init__(self, label, confidence):
                self.label = label
                self.confidence = confidence

        class _Result:
            pass

        r = _Result()
        r.damage_types = [_Pred(self.damage_type, 0.91)]
        r.severity = _Pred(self.severity, 0.84)
        r.low_confidence = False
        r.model_version = "injected_v1"
        r.source_image = str(image_path)
        r.timestamp = "2026-02-15T00:00:00"
        r.error = None
        return r


def _make_claim(db_session, *, image_filename: str = "rear.jpg"):
    cust = Customer(name="Inj", email="inj@test.com", phone="0")
    db_session.add(cust); db_session.flush()
    veh = Vehicle(customer_id=cust.id, make="Honda", model="Civic",
                  year=2020, vin="INJVIN", plate_number="INJ-1")
    db_session.add(veh); db_session.flush()
    pol = Policy(customer_id=cust.id, vehicle_id=veh.id,
                 policy_number="INJPOL", coverage_type="comprehensive",
                 coverage_limit=50000, deductible=500,
                 start_date=__import__("datetime").date(2025, 1, 1),
                 end_date=__import__("datetime").date(2026, 12, 31),
                 status="active")
    db_session.add(pol); db_session.flush()
    claim = Claim(claim_number="INJ-CLM", policy_id=pol.id,
                  vehicle_id=veh.id,
                  incident_date=__import__("datetime").date(2026, 2, 15),
                  reported_date=__import__("datetime").date(2026, 2, 16),
                  claimed_amount=1000.0, status="pending")
    db_session.add(claim); db_session.flush()
    region_ref = json.dumps({"image_path": f"uploads/{claim.id}/{image_filename}"})
    dmg = Damage(claim_id=claim.id, source="image",
                 damage_type="pending", severity="pending", confidence=None,
                 region_ref=region_ref)
    db_session.add(dmg); db_session.flush()
    # Write a tiny placeholder file at the expected path.
    from app.core.config import settings
    import os
    base = Path(settings.upload_dir)
    if not base.is_absolute():
        base = Path(os.getcwd()) / base
    claim_dir = base / str(claim.id)
    claim_dir.mkdir(parents=True, exist_ok=True)
    (claim_dir / image_filename).write_bytes(
        b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"
    )
    return claim


def test_injected_predictor_is_used_when_ml_package_missing(
    db_session, monkeypatch
):
    """When `ml` is not on sys.path, the injected predictor must still
    drive the run — the function must NOT import the real package.
    """
    # Snapshot of sys.modules, then drop anything `ml.*` so the import
    # would fail if attempted.
    saved = {k: v for k, v in sys.modules.items() if k == "ml" or k.startswith("ml.")}
    for k in list(sys.modules):
        if k == "ml" or k.startswith("ml."):
            del sys.modules[k]
    monkeypatch.setitem(sys.modules, "ml", None)  # forces ImportError
    # Note: setting sys.modules["ml"] = None is treated by Python as
    # "the module exists but is None" — it does NOT raise ImportError.
    # Use the `del` path above to actually remove it.

    claim = _make_claim(db_session)
    dmg = db_session.query(Damage).filter(
        Damage.claim_id == claim.id, Damage.source == "image",
    ).first()
    predictor = _InjectedPredictor("scratch", "minor")

    new_rows = cv_service.run_cv_on_image(
        db_session, claim.id, dmg.region_ref, predictor=predictor,
    )

    # The injected predictor was used.
    assert len(new_rows) == 1
    assert new_rows[0].damage_type == "scratch"
    assert new_rows[0].severity == "minor"
    assert len(predictor.calls) == 1


def test_injected_predictor_does_not_crash_when_ml_unimportable(
    db_session, monkeypatch
):
    """If `ml` is genuinely unimportable and no predictor is supplied,
    the function must gracefully fall back to a `cv_error` row rather
    than propagating the ImportError (which would have been the
    pre-Phase-13 behavior).
    """
    for k in list(sys.modules):
        if k == "ml" or k.startswith("ml."):
            del sys.modules[k]
    # Make `ml` raise ImportError when imported.
    import importlib
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "ml.inference.predictor":
            raise ImportError("ml package deliberately hidden for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    claim = _make_claim(db_session, image_filename="front.jpg")
    dmg = db_session.query(Damage).filter(
        Damage.claim_id == claim.id, Damage.source == "image",
    ).first()

    new_rows = cv_service.run_cv_on_image(
        db_session, claim.id, dmg.region_ref, predictor=None,
    )
    assert len(new_rows) == 1
    assert new_rows[0].damage_type == "cv_error"
    assert new_rows[0].severity == "unknown"
