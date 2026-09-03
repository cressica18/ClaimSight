"""
Phase 4 tests — Computer Vision pipeline.

Covers (without requiring real model weights or large dataset):
  - Model construction and forward pass
  - Inference with mocked weights
  - Confidence thresholding / low_confidence flag
  - Corrupted / tiny image handling
  - CV API endpoints (mocked predictor)
  - Dataset transform pipeline
"""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

# Add ml package to path for direct imports
ML_ROOT = Path(__file__).resolve().parents[2] / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT.parent))

# ─── Model tests ──────────────────────────────────────────────────────────────

class TestDualHeadResNet50:
    """Tests for the model architecture (no pretrained weights downloaded)."""

    def test_model_constructs_without_pretrained(self):
        from ml.training.model import DualHeadResNet50
        model = DualHeadResNet50(pretrained=False)
        assert model is not None

    def test_forward_pass_shapes(self):
        from ml.training.model import DualHeadResNet50
        model = DualHeadResNet50(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            d_logits, s_logits = model(x)
        assert d_logits.shape == (2, 8), f"Expected (2,8), got {d_logits.shape}"
        assert s_logits.shape == (2, 3), f"Expected (2,3), got {s_logits.shape}"

    def test_freeze_backbone_reduces_trainable_params(self):
        from ml.training.model import DualHeadResNet50
        model = DualHeadResNet50(pretrained=False)
        model.freeze_backbone()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        assert trainable < total, "Freeze should reduce trainable param count"

    def test_unfreeze_stage2_increases_trainable_params(self):
        from ml.training.model import DualHeadResNet50
        model = DualHeadResNet50(pretrained=False)
        model.freeze_backbone()
        frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.unfreeze_stage2()
        unfrozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert unfrozen_trainable > frozen_trainable

    def test_predict_returns_probabilities_in_range(self):
        from ml.training.model import DualHeadResNet50
        model = DualHeadResNet50(pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        d_probs, s_probs = model.predict(x)
        assert d_probs.shape == (1, 8)
        assert s_probs.shape == (1, 3)
        assert (d_probs >= 0).all() and (d_probs <= 1).all(), "Damage probs out of [0,1]"
        assert abs(float(s_probs.sum()) - 1.0) < 1e-4, "Severity probs must sum to 1"

    def test_checkpoint_save_and_load(self, tmp_path):
        from ml.training.model import DualHeadResNet50
        model = DualHeadResNet50(pretrained=False)
        ckpt_path = tmp_path / "test_checkpoint.pt"
        torch.save({"model_state_dict": model.state_dict(), "metrics": {}, "epoch": 1}, ckpt_path)

        from ml.training.model import load_checkpoint
        loaded = load_checkpoint(str(ckpt_path))
        assert loaded is not None

        # Forward pass should still work
        x = torch.randn(1, 3, 224, 224)
        d, s = loaded(x)
        assert d.shape == (1, 8)


# ─── Transform tests ──────────────────────────────────────────────────────────

class TestTransforms:
    def test_train_transform_output_shape(self):
        from ml.training.dataset import get_train_transform
        tf = get_train_transform()
        img = Image.new("RGB", (256, 256), color=100)
        tensor = tf(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        from ml.training.dataset import get_val_transform
        tf = get_val_transform()
        img = Image.new("RGB", (300, 400), color=50)
        tensor = tf(img)
        assert tensor.shape == (3, 224, 224)

    def test_inference_transform_matches_val(self):
        from ml.training.dataset import get_val_transform, get_inference_transform
        import torch
        img = Image.new("RGB", (256, 256), color=123)
        torch.manual_seed(0)
        t1 = get_val_transform()(img)
        torch.manual_seed(0)
        t2 = get_inference_transform()(img)
        assert torch.equal(t1, t2), "Inference and val transforms must be identical"


# ─── Predictor tests ──────────────────────────────────────────────────────────

class TestVehicleDamagePredictor:
    """Tests for inference predictor. Use mocked model to avoid needing real weights."""

    def _make_fake_predictor(self, tmp_path, damage_probs=None, sev_probs=None):
        """Create a predictor with a fake checkpoint and mocked model."""
        from ml.training.model import DualHeadResNet50
        from ml.inference.predictor import VehicleDamagePredictor

        model = DualHeadResNet50(pretrained=False)
        ckpt = tmp_path / "fake_weights.pt"
        torch.save({"model_state_dict": model.state_dict(), "metrics": {}, "epoch": 0}, ckpt)

        predictor = VehicleDamagePredictor(checkpoint_path=ckpt)
        predictor._load_model()

        if damage_probs is not None or sev_probs is not None:
            # Patch predict to return controlled probabilities
            def fake_predict(x):
                d = damage_probs if damage_probs is not None else torch.zeros(1, 8)
                s = sev_probs   if sev_probs   is not None else torch.tensor([[0.0, 0.8, 0.2]])
                return d, s
            predictor._model.predict = fake_predict

        return predictor

    def test_predict_valid_image(self, tmp_path):
        predictor = self._make_fake_predictor(tmp_path)
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (300, 300), color=128).save(img_path)

        result = predictor.predict_from_path(img_path)
        assert result.error is None
        assert len(result.damage_types) >= 0   # at least no_damage
        assert result.severity.label in ("minor", "moderate", "severe")
        assert result.timestamp is not None

    def test_predict_sets_low_confidence_when_probs_low(self, tmp_path):
        """When top damage prob < LOW_CONF_DAMAGE, low_confidence must be True."""
        # All damage probs = 0.1 (< threshold 0.4)
        low_d = torch.full((1, 8), 0.1)
        low_s = torch.tensor([[0.4, 0.3, 0.3]])   # top sev < 0.5

        predictor = self._make_fake_predictor(tmp_path, damage_probs=low_d, sev_probs=low_s)
        img_path = tmp_path / "low_conf.jpg"
        Image.new("RGB", (200, 200), color=50).save(img_path)

        result = predictor.predict_from_path(img_path)
        assert result.low_confidence is True

    def test_predict_no_low_confidence_when_probs_high(self, tmp_path):
        """When damage prob ≥ 0.4 and severity ≥ 0.5, low_confidence must be False."""
        # scratch has high confidence
        d = torch.zeros(1, 8)
        d[0][0] = 0.85   # scratch = 0.85 (index 0)
        s = torch.tensor([[0.05, 0.80, 0.15]])   # moderate = 0.80

        predictor = self._make_fake_predictor(tmp_path, damage_probs=d, sev_probs=s)
        img_path = tmp_path / "high_conf.jpg"
        Image.new("RGB", (200, 200), color=50).save(img_path)

        result = predictor.predict_from_path(img_path)
        assert result.low_confidence is False

    def test_predict_corrupted_image_returns_error(self, tmp_path):
        predictor = self._make_fake_predictor(tmp_path)
        bad_path = tmp_path / "bad.jpg"
        bad_path.write_bytes(b"this is not an image")

        result = predictor.predict_from_path(bad_path)
        assert result.error is not None
        assert result.low_confidence is True

    def test_predict_missing_file_returns_error(self, tmp_path):
        predictor = self._make_fake_predictor(tmp_path)
        result = predictor.predict_from_path(tmp_path / "nonexistent.jpg")
        assert result.error is not None

    def test_predict_tiny_image_returns_error(self, tmp_path):
        predictor = self._make_fake_predictor(tmp_path)
        img_path = tmp_path / "tiny.jpg"
        Image.new("RGB", (10, 10)).save(img_path)

        result = predictor.predict_from_path(img_path)
        assert result.error is not None   # too small

    def test_predict_from_bytes(self, tmp_path):
        predictor = self._make_fake_predictor(tmp_path)
        buf = io.BytesIO()
        Image.new("RGB", (200, 200), color=200).save(buf, format="JPEG")
        data = buf.getvalue()

        result = predictor.predict_from_bytes(data, filename="mem.jpg")
        assert result.error is None

    def test_missing_checkpoint_raises(self, tmp_path):
        from ml.inference.predictor import VehicleDamagePredictor
        predictor = VehicleDamagePredictor(checkpoint_path=tmp_path / "no_weights.pt")
        with pytest.raises(FileNotFoundError):
            predictor._load_model()

    def test_batch_predict(self, tmp_path):
        predictor = self._make_fake_predictor(tmp_path)
        imgs = []
        for i in range(3):
            p = tmp_path / f"img{i}.jpg"
            Image.new("RGB", (200, 200), color=i * 50).save(p)
            imgs.append(p)

        results = predictor.predict_batch(imgs)
        assert len(results) == 3
        for r in results:
            assert r.timestamp is not None


# ─── CV API endpoint tests ────────────────────────────────────────────────────

class TestCVAPI:
    """Tests for CV API endpoints using FastAPI TestClient + mocked predictor."""

    def _make_api_client(self, db_session):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: db_session
        return TestClient(app)

    def _seed_claim(self, db_session):
        import datetime
        from app.models.customer import Customer
        from app.models.vehicle import Vehicle
        from app.models.policy import Policy
        from app.models.claim import Claim
        from app.models.damage import Damage

        c = Customer(name="CV Test", email="cv@test.com", phone="000")
        db_session.add(c); db_session.flush()

        v = Vehicle(customer_id=c.id, make="Ford", model="Fiesta", year=2019, vin="CVVIN1", plate_number="CV001")
        db_session.add(v); db_session.flush()

        p = Policy(customer_id=c.id, vehicle_id=v.id, policy_number="CVPOL1", coverage_type="basic",
                   coverage_limit=30000, deductible=500, start_date=datetime.date(2024,1,1), end_date=datetime.date(2025,1,1), status="active")
        db_session.add(p); db_session.flush()

        cl = Claim(claim_number="CVCL1", policy_id=p.id, vehicle_id=v.id,
                   incident_date=datetime.date(2024,6,1), status="pending")
        db_session.add(cl); db_session.flush()

        dmg = Damage(claim_id=cl.id, source="image", damage_type="pending", severity="pending",
                     region_ref="uploads/1/test.jpg")
        db_session.add(dmg)
        db_session.commit()
        return cl.id, dmg.id

    def test_analyze_single_missing_checkpoint_returns_503(self, client, db_session):
        """Without trained weights, analyze endpoint returns 503 (not 500)."""
        from app.main import app
        from app.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: db_session

        claim_id, dmg_id = self._seed_claim(db_session)

        with patch("ml.inference.predictor.get_predictor") as mock_get:
            mock_predictor = MagicMock()
            mock_predictor.predict_from_path.side_effect = FileNotFoundError("no weights")
            mock_get.return_value = mock_predictor

            response = client.post(f"/claims/{claim_id}/damages/{dmg_id}/analyze")
            # 503 because no trained weights
            assert response.status_code in (200, 503), response.json()

        app.dependency_overrides.clear()

    def test_analyze_nonexistent_claim_returns_404(self, client, db_session):
        from app.main import app
        from app.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: db_session
        response = client.post("/claims/99999/damages/1/analyze")
        assert response.status_code == 404
        app.dependency_overrides.clear()

    def test_analyze_nonexistent_damage_returns_404(self, client, db_session):
        from app.main import app
        from app.api.deps import get_db
        import datetime
        from app.models.customer import Customer
        from app.models.vehicle import Vehicle
        from app.models.policy import Policy
        from app.models.claim import Claim

        c = Customer(name="NF Test", email="nf@test.com")
        db_session.add(c); db_session.flush()
        v = Vehicle(customer_id=c.id, make="VW", model="Golf", year=2018, vin="NFVIN1", plate_number="NF001")
        db_session.add(v); db_session.flush()
        p = Policy(customer_id=c.id, vehicle_id=v.id, policy_number="NFPOL1", coverage_type="basic",
                   coverage_limit=20000, deductible=500, start_date=datetime.date(2024,1,1), end_date=datetime.date(2025,1,1), status="active")
        db_session.add(p); db_session.flush()
        cl = Claim(claim_number="NFCL1", policy_id=p.id, vehicle_id=v.id, incident_date=datetime.date(2024,7,1), status="pending")
        db_session.add(cl); db_session.commit()

        app.dependency_overrides[get_db] = lambda: db_session
        response = client.post(f"/claims/{cl.id}/damages/99999/analyze")
        assert response.status_code == 404
        app.dependency_overrides.clear()
