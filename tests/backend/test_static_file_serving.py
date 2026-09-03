"""Tests for the static /api/uploads/* file serving path.

The frontend's Document Viewer "Open file in new tab" link hits
`/api/{file_path}` where `file_path` starts with `uploads/`. The
FastAPI app mounts a `StaticFiles` directory at `/api/uploads` so
the request is served from the on-disk upload directory. The
response must include `Content-Type: application/pdf` for PDFs so
the browser's built-in viewer takes over.

Regression test for the Phase 14 bug-fix pass: a valid PDF must
return 200 with the PDF content type.

The test writes the PDF into the real upload directory (the one
the app's `StaticFiles` mount is bound to) and removes it
afterward so other tests are not affected.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# A small but valid PDF — single blank page, opens cleanly in
# Chrome's built-in viewer. The earlier demo generator's 21-byte
# stub (`%PDF-1.4\n% demo stub\n`) was missing the xref table and
# %%EOF marker, so browsers refused it with "Failed to load PDF
# document." This byte string is the same one the demo generator
# now writes, so the round-trip is the same path the live app
# exercises.
VALID_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
    b"xref\n0 4\n0000000000 65535 f \n"
    b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n183\n%%EOF\n"
)


def _upload_base() -> Path:
    """Resolve the same path `app.main` did at import time."""
    from app.core.config import settings
    base = Path(settings.upload_dir)
    if not base.is_absolute():
        base = Path(os.getcwd()) / base
    return base


@pytest.fixture
def pdf_on_disk():
    """Place a valid PDF in a subdirectory of the real upload dir
    and remove it on teardown. The app's `StaticFiles` mount is
    already bound to this directory, so this is the only way to
    make the test exercise the live path.
    """
    base = _upload_base()
    claim_dir = base / "_test_"
    claim_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = claim_dir / "claim-form.pdf"
    pdf_path.write_bytes(VALID_PDF_BYTES)
    try:
        yield pdf_path
    finally:
        try:
            pdf_path.unlink()
        except FileNotFoundError:
            pass
        try:
            claim_dir.rmdir()
        except OSError:
            pass  # Directory not empty or already removed


def test_valid_pdf_is_served_with_correct_content_type(
    pdf_on_disk, client: TestClient
) -> None:
    """A valid PDF stored under `uploads/{claim_id}/foo.pdf` must
    return 200 with `Content-Type: application/pdf` and the exact
    bytes that were written. This is the path the Document Viewer
    "Open file in new tab" link takes.
    """
    response = client.get("/api/uploads/_test_/claim-form.pdf")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == VALID_PDF_BYTES


def test_missing_pdf_returns_404(client: TestClient) -> None:
    """Asking for a file that does not exist on disk must 404
    rather than silently returning an empty body or a corrupt
    file. The browser's PDF viewer depends on this so it can
    surface "this document is missing" instead of "corrupt PDF".
    """
    response = client.get("/api/uploads/_test_/no-such-file.pdf")
    assert response.status_code == 404
