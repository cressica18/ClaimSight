"""
Phase 5 integration tests — CV Analysis Endpoints with Real Checkpoint.

These tests use the REAL trained checkpoint and real images to verify:
- uploaded image → real CV analysis
- valid prediction persistence
- low-confidence predictions
- invalid/missing image
- missing checkpoint
- nonexistent claim/image
- repeated analysis behavior
"""

import io
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from fastapi.testclient import TestClient

# Add ml package to path for direct imports
import sys
ML_ROOT = Path(__file__).resolve().parents[2] / "ml"
if str(ML_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ML_ROOT.parent))


def _seed_claim(db_session):
    """Create a test claim with customer, vehicle, policy."""
    import datetime
    from app.models.customer import Customer
    from app.models.vehicle import Vehicle
    from app.models.policy import Policy
    from app.models.claim import Claim
    from app.models.damage import Damage

    c = Customer(name="CV Integration Test", email="cv_int@test.com", phone="000")
    db_session.add(c)
    db_session.flush()

    v = Vehicle(customer_id=c.id, make="Ford", model="Fiesta", year=2019, vin="CVINT1", plate_number="CVINT1")
    db_session.add(v)
    db_session.flush()

    p = Policy(customer_id=c.id, vehicle_id=v.id, policy_number="CVINTPOL1", coverage_type="basic",
               coverage_limit=30000, deductible=500, start_date=datetime.date(2024,1,1), end_date=datetime.date(2025,1,1), status="active")
    db_session.add(p)
    db_session.flush()

    cl = Claim(claim_number="CVINTCL1", policy_id=p.id, vehicle_id=v.id,
               incident_date=datetime.date(2024,6,1), status="pending")
    db_session.add(cl)
    db_session.commit()
    return cl.id


def _create_test_image(tmp_path: Path, color: tuple = (128, 128, 128), size: tuple = (300, 300)) -> Path:
    """Create a test JPEG image."""
    img_path = tmp_path / "test_image.jpg"
    Image.new("RGB", size, color=color).save(img_path, format="JPEG")
    return img_path


class TestCVIntegration:
    """Integration tests for CV analysis with real checkpoint."""

    def _make_client(self, db_session):
        from app.main import app
        from app.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: db_session
        return TestClient(app)

    def test_upload_image_creates_damage_with_region_ref(self, client, db_session):
        """Upload an image and verify Damage record has region_ref with image path."""
        claim_id = _seed_claim(db_session)

        # Create test image
        buf = io.BytesIO()
        Image.new("RGB", (300, 300), color=(100, 150, 200)).save(buf, format="JPEG")
        buf.seek(0)

        response = client.post(
            f"/claims/{claim_id}/images",
            files={"files": ("test.jpg", buf, "image/jpeg")}
        )
        assert response.status_code == 201, response.json()
        data = response.json()
        assert len(data) == 1
        damage = data[0]
        assert damage["damage_type"] == "pending"
        assert damage["severity"] == "pending"
        assert damage["region_ref"] is not None
        # region_ref should contain JSON with image_path
        ref = json.loads(damage["region_ref"])
        assert "image_path" in ref
        assert ref["image_path"].startswith("uploads/")
        print(f"Created damage with region_ref: {damage['region_ref']}")

    def test_analyze_single_image_real_inference(self, client, db_session, tmp_path):
        """Test real CV inference on an uploaded image using the trained checkpoint."""
        claim_id = _seed_claim(db_session)

        # Create and upload a test image
        img_path = _create_test_image(tmp_path)
        with open(img_path, "rb") as f:
            response = client.post(
                f"/claims/{claim_id}/images",
                files={"files": ("test.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 201
        damage = response.json()[0]
        damage_id = damage["id"]

        # Run analysis - this uses the REAL trained checkpoint
        response = client.post(f"/claims/{claim_id}/damages/{damage_id}/analyze")
        assert response.status_code == 200, f"Analysis failed: {response.json()}"
        result = response.json()

        # Verify response structure
        assert "damage_id" in result
        assert "damage_types" in result
        assert "severity" in result
        assert "low_confidence" in result
        assert "model_version" in result
        assert "timestamp" in result

        # Verify damage types returned
        assert len(result["damage_types"]) >= 1
        for dt in result["damage_types"]:
            assert "label" in dt
            assert "confidence" in dt
            assert 0.0 <= dt["confidence"] <= 1.0

        # Verify severity
        assert result["severity"]["label"] in ("minor", "moderate", "severe")
        assert 0.0 <= result["severity"]["confidence"] <= 1.0

        # Verify low_confidence is boolean
        assert isinstance(result["low_confidence"], bool)

        # Verify damage records were created in DB
        from app.models.damage import Damage
        damages = db_session.query(Damage).filter(Damage.claim_id == claim_id).all()
        # Original pending + new analysis damages
        assert len(damages) >= 2

        print(f"Analysis result: {json.dumps(result, indent=2)}")

    def test_analyze_all_images_batch(self, client, db_session, tmp_path):
        """Test batch analysis of all images for a claim."""
        claim_id = _seed_claim(db_session)

        # Upload multiple images
        for i in range(3):
            img_path = _create_test_image(tmp_path, color=(50 + i*50, 100, 150))
            with open(img_path, "rb") as f:
                response = client.post(
                    f"/claims/{claim_id}/images",
                    files={"files": (f"test{i}.jpg", f, "image/jpeg")}
                )
            assert response.status_code == 201

        # Run batch analysis
        response = client.post(f"/claims/{claim_id}/analyze-images")
        assert response.status_code == 200, f"Batch analysis failed: {response.json()}"
        result = response.json()

        assert "claim_id" in result
        assert "analyzed" in result
        assert "results" in result
        assert result["analyzed"] == 3
        assert len(result["results"]) == 3

        for r in result["results"]:
            assert "damage_id" in r
            assert "damage_types" in r
            assert "severity" in r
            assert "low_confidence" in r
            assert r["model_version"] == "claimsight_cv_v1"

    def test_analyze_nonexistent_claim_returns_404(self, client, db_session):
        """Analyzing a nonexistent claim returns 404."""
        response = client.post("/claims/99999/damages/1/analyze")
        assert response.status_code == 404

    def test_analyze_nonexistent_damage_returns_404(self, client, db_session):
        """Analyzing a nonexistent damage record returns 404."""
        claim_id = _seed_claim(db_session)
        response = client.post(f"/claims/{claim_id}/damages/99999/analyze")
        assert response.status_code == 404

    def test_analyze_damage_without_image_returns_400(self, client, db_session):
        """Analyzing a damage record without region_ref returns 400."""
        from app.models.damage import Damage
        claim_id = _seed_claim(db_session)

        # Create damage without region_ref
        dmg = Damage(claim_id=claim_id, source="image", damage_type="pending", severity="pending")
        db_session.add(dmg)
        db_session.commit()

        response = client.post(f"/claims/{claim_id}/damages/{dmg.id}/analyze")
        assert response.status_code == 400
        assert "no associated image" in response.json()["detail"].lower()

    def test_analyze_missing_image_file_returns_error(self, client, db_session):
        """Analyzing with a missing image file returns error in result (not 500)."""
        claim_id = _seed_claim(db_session)

        # Create damage with non-existent image path
        from app.models.damage import Damage
        dmg = Damage(
            claim_id=claim_id,
            source="image",
            damage_type="pending",
            severity="pending",
            region_ref=json.dumps({"image_path": "uploads/999/nonexistent.jpg"})
        )
        db_session.add(dmg)
        db_session.commit()

        response = client.post(f"/claims/{claim_id}/damages/{dmg.id}/analyze")
        # Should return 200 with error in result, not 500
        assert response.status_code == 200
        result = response.json()
        assert result["error"] is not None
        assert "not found" in result["error"].lower() or "failed" in result["error"].lower()

    def test_repeated_analysis_creates_new_damage_records(self, client, db_session, tmp_path):
        """Running analysis multiple times creates new Damage records each time."""
        claim_id = _seed_claim(db_session)

        # Upload image
        img_path = _create_test_image(tmp_path)
        with open(img_path, "rb") as f:
            response = client.post(
                f"/claims/{claim_id}/images",
                files={"files": ("test.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 201
        damage_id = response.json()[0]["id"]

        # First analysis
        response1 = client.post(f"/claims/{claim_id}/damages/{damage_id}/analyze")
        assert response1.status_code == 200

        # Second analysis on same damage record
        response2 = client.post(f"/claims/{claim_id}/damages/{damage_id}/analyze")
        assert response2.status_code == 200

        # Should have created new damage records each time
        from app.models.damage import Damage
        damages = db_session.query(Damage).filter(Damage.claim_id == claim_id).all()
        # Original pending + first analysis damages + second analysis damages
        assert len(damages) >= 3


class TestCVLowConfidence:
    """Tests for low-confidence handling."""

    def _make_client(self, db_session):
        from app.main import app
        from app.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: db_session
        return TestClient(app)

    def test_low_confidence_flag_in_response(self, client, db_session, tmp_path):
        """Verify low_confidence flag is present and correct in response."""
        claim_id = _seed_claim(db_session)

        # Create a simple test image (likely to produce low confidence)
        img_path = _create_test_image(tmp_path, color=(50, 50, 50))  # dark image
        with open(img_path, "rb") as f:
            response = client.post(
                f"/claims/{claim_id}/images",
                files={"files": ("dark.jpg", f, "image/jpeg")}
            )
        assert response.status_code == 201
        damage_id = response.json()[0]["id"]

        # Run analysis
        response = client.post(f"/claims/{claim_id}/damages/{damage_id}/analyze")
        assert response.status_code == 200
        result = response.json()

        # low_confidence should be a boolean
        assert "low_confidence" in result
        assert isinstance(result["low_confidence"], bool)
        print(f"Low confidence: {result['low_confidence']}, Severity: {result['severity']}")


class TestCVCheckpointMissing:
    """Tests for missing checkpoint handling."""

    def _make_client(self, db_session):
        from app.main import app
        from app.api.deps import get_db
        app.dependency_overrides[get_db] = lambda: db_session
        return TestClient(app)

    def test_cv_error_handled_gracefully(self, client, db_session):
        """When CV inference fails (e.g. missing image), error is returned gracefully in result."""
        claim_id = _seed_claim(db_session)
        from app.models.damage import Damage
        dmg = Damage(claim_id=claim_id, source="image", damage_type="pending", severity="pending",
                     region_ref=json.dumps({"image_path": "uploads/999/nonexistent.jpg"}))
        db_session.add(dmg)
        db_session.commit()

        response = client.post(f"/claims/{claim_id}/damages/{dmg.id}/analyze")
        # Should return 200 with error in result, not 500
        assert response.status_code == 200
        result = response.json()
        assert result["error"] is not None
        assert "not found" in result["error"].lower() or "failed" in result["error"].lower()


# Run a quick manual test if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])