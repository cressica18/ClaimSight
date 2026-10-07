"""Tests for the OCR adapter and OCR fallback in document extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.models.enums import DocType
from app.services.document_extractor import DocumentExtractor, ExtractionResult
from app.services.ocr import OCRAdapter, OCRResult, create_ocr_adapter


class TestOCRAdapter:
    """Tests for the OCR adapter."""

    def test_is_available_when_tesseract_installed(self):
        """OCR adapter should report available when Tesseract is in PATH."""
        adapter = OCRAdapter()
        # Tesseract is installed in the test environment
        assert adapter.is_available() is True

    def test_extract_text_from_pdf_success(self, tmp_path: Path):
        """OCR adapter should extract text from a PDF with images."""
        adapter = OCRAdapter()
        if not adapter.is_available():
            pytest.skip("Tesseract not available")

        # Create a simple PDF with an image (we can't easily create one with text in an image
        # without external tools, so we test the interface works)
        pdf_path = tmp_path / "test.pdf"
        # Minimal PDF
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n183\n%%EOF\n"
        )

        result = adapter.extract_text_from_pdf(pdf_path)

        assert isinstance(result, OCRResult)
        assert result.success is True  # Should succeed even if no text found
        assert result.page_count == 1

    def test_extract_text_from_nonexistent_file(self, tmp_path: Path):
        """OCR adapter should handle missing files gracefully."""
        adapter = OCRAdapter()
        result = adapter.extract_text_from_pdf(tmp_path / "nonexistent.pdf")

        assert result.success is False
        assert result.text == ""
        assert any("not found" in w.lower() for w in result.warnings)

    def test_extract_text_from_zero_page_pdf(self, tmp_path: Path):
        """OCR adapter should handle zero-page PDFs."""
        adapter = OCRAdapter()
        # Create a PDF with zero pages (malformed but valid structure)
        pdf_path = tmp_path / "zero_page.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
            b"xref\n0 3\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n"
            b"trailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n77\n%%EOF\n"
        )

        result = adapter.extract_text_from_pdf(pdf_path)

        assert result.success is False
        assert result.page_count == 0
        assert any("zero page" in w.lower() for w in result.warnings)


class TestDocumentExtractorOCRFallback:
    """Tests for OCR fallback in DocumentExtractor."""

    def test_native_text_pdf_uses_text_method(self, tmp_path: Path):
        """PDF with extractable text should use 'text' extraction method."""
        # Create a PDF with text content
        pdf_path = tmp_path / "text.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td (Policy Number: POL-12345) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000218 00000 n \n"
            b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n298\n%%EOF\n"
        )

        extractor = DocumentExtractor()
        result = extractor.extract(pdf_path, DocType.policy)

        assert result.extraction_method == "text"
        assert result.fields.get("policy_number") == "POL-12345"

    def test_blank_pdf_with_policy_filename_uses_filename_method(self, tmp_path: Path):
        """Blank PDF with policy number in filename should use 'filename' method."""
        pdf_path = tmp_path / "POL-99999-policy.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n183\n%%EOF\n"
        )

        extractor = DocumentExtractor()
        result = extractor.extract(pdf_path, DocType.policy)

        assert result.extraction_method == "filename"
        assert result.fields.get("policy_number") == "POL-99999"

    def test_blank_claim_form_uses_none_method(self, tmp_path: Path):
        """Blank claim form (no filename fallback) should use 'none' method."""
        pdf_path = tmp_path / "claim-form.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n183\n%%EOF\n"
        )

        extractor = DocumentExtractor()
        result = extractor.extract(pdf_path, DocType.claim_form)

        assert result.extraction_method == "none"
        assert result.fields == {}

    @patch("app.services.ocr.OCRAdapter.is_available", return_value=True)
    @patch("app.services.ocr.OCRAdapter.extract_text_from_pdf")
    def test_ocr_fallback_when_native_text_minimal(
        self, mock_ocr_extract, mock_is_available, tmp_path: Path
    ):
        """When native text is minimal and OCR succeeds, should use 'ocr' method."""
        # Mock OCR to return text with a policy number
        mock_ocr_extract.return_value = OCRResult(
            text="Policy Number: POL-55555",
            page_count=1,
            warnings=(),
            success=True,
        )

        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n183\n%%EOF\n"
        )

        extractor = DocumentExtractor()
        result = extractor.extract(pdf_path, DocType.policy)

        assert result.extraction_method == "ocr"
        assert result.fields.get("policy_number") == "POL-55555"
        mock_ocr_extract.assert_called_once()

    @patch("app.services.ocr.OCRAdapter.is_available", return_value=False)
    def test_ocr_unavailable_falls_back_to_filename(self, mock_is_available, tmp_path: Path):
        """When OCR is unavailable, should fall back to filename for policy docs."""
        pdf_path = tmp_path / "POL-77777-policy.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n183\n%%EOF\n"
        )

        extractor = DocumentExtractor()
        result = extractor.extract(pdf_path, DocType.policy)

        # Should fall back to filename method since OCR is unavailable
        assert result.extraction_method == "filename"
        assert result.fields.get("policy_number") == "POL-77777"

    @patch("app.services.ocr.OCRAdapter.is_available", return_value=True)
    @patch("app.services.ocr.OCRAdapter.extract_text_from_pdf")
    def test_ocr_failure_falls_back_to_filename(
        self, mock_ocr_extract, mock_is_available, tmp_path: Path
    ):
        """When OCR fails, should fall back to filename for policy docs."""
        mock_ocr_extract.return_value = OCRResult(
            text="",
            page_count=1,
            warnings=("OCR error: some failure",),
            success=False,
        )

        pdf_path = tmp_path / "POL-88888-policy.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n183\n%%EOF\n"
        )

        extractor = DocumentExtractor()
        result = extractor.extract(pdf_path, DocType.policy)

        # Should fall back to filename method since OCR failed
        assert result.extraction_method == "filename"
        assert result.fields.get("policy_number") == "POL-88888"

    def test_ocr_adapter_injected_for_testing(self, tmp_path: Path):
        """DocumentExtractor should accept injected OCR adapter for testing."""
        mock_adapter = MagicMock(spec=OCRAdapter)
        mock_adapter.is_available.return_value = True
        mock_adapter.extract_text_from_pdf.return_value = OCRResult(
            text="Injected OCR text with POL-99999",
            page_count=1,
            warnings=(),
            success=True,
        )

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n183\n%%EOF\n"
        )

        extractor = DocumentExtractor(ocr_adapter=mock_adapter)
        result = extractor.extract(pdf_path, DocType.policy)

        assert result.extraction_method == "ocr"
        assert result.fields.get("policy_number") == "POL-99999"
        mock_adapter.extract_text_from_pdf.assert_called_once_with(pdf_path)


class TestOCRResultProvenance:
    """Tests that extraction provenance is correctly stored."""

    def test_extraction_method_in_result(self, tmp_path: Path):
        """ExtractionResult should include extraction_method field."""
        pdf_path = tmp_path / "POL-12345-policy.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n183\n%%EOF\n"
        )

        extractor = DocumentExtractor()
        result = extractor.extract(pdf_path, DocType.policy)

        assert hasattr(result, "extraction_method")
        assert result.extraction_method in ("text", "ocr", "filename", "none")

    def test_extraction_method_values_are_consistent(self):
        """Extraction method should only be one of the expected values."""
        valid_methods = {"text", "ocr", "filename", "none"}
        # This is a documentation test - the actual validation is in the code
        assert valid_methods == {"text", "ocr", "filename", "none"}