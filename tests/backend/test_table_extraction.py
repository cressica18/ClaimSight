"""Tests for table extraction in repair estimates and invoices."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.table_extractor import (
    RepairLineItem,
    TableExtractionResult,
    _extract_line_items_from_text,
    _is_header_line,
    _parse_complex_line,
    _parse_qty_unit_line,
    _parse_simple_line,
    extract_repair_tables,
    _SIMPLE_LINE_ITEM_PATTERN,
    _QTY_UNIT_TOTAL_PATTERN,
)


class TestTableExtraction:
    """Tests for the table extraction functionality."""

    def test_extract_repair_tables_text_fallback(self, tmp_path: Path):
        """Test table extraction with text-based fallback."""
        # Create a PDF with text that looks like a table
        pdf_path = tmp_path / "estimate.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 100 >>\nstream\nBT /F1 12 Tf 100 700 Td (REPAIR ESTIMATE) Tj ET\nBT /F1 12 Tf 100 680 Td (front bumper - replace - $50000.00) Tj ET\nBT /F1 12 Tf 100 660 Td (rear quarter panel - repair - $400.00) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000218 00000 n \n"
            b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n298\n%%EOF\n"
        )

        result = extract_repair_tables(pdf_path)

        assert result.success is True
        assert len(result.line_items) == 2
        assert result.line_items[0].part_name == "front bumper"
        assert result.line_items[0].operation == "replace"
        assert result.line_items[0].line_total == 50000.0
        assert result.line_items[1].part_name == "rear quarter panel"
        assert result.line_items[1].operation == "repair"
        assert result.line_items[1].line_total == 400.0

    def test_extract_repair_tables_with_qty_unit_total(self, tmp_path: Path):
        """Test extraction with quantity, unit price, and total format."""
        pdf_path = tmp_path / "estimate.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 120 >>\nstream\nBT /F1 12 Tf 100 700 Td (REPAIR ESTIMATE) Tj ET\nBT /F1 12 Tf 100 680 Td (Front Bumper  1  35000.00  35000.00) Tj ET\nBT /F1 12 Tf 100 660 Td (Paint  2  150.00  300.00) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000218 00000 n \n"
            b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n298\n%%EOF\n"
        )

        result = extract_repair_tables(pdf_path)

        assert result.success is True
        assert len(result.line_items) == 2
        assert result.line_items[0].part_name == "Front Bumper"
        assert result.line_items[0].quantity == 1.0
        assert result.line_items[0].unit_price == 35000.0
        assert result.line_items[0].line_total == 35000.0
        assert result.line_items[1].part_name == "Paint"
        assert result.line_items[1].quantity == 2.0
        assert result.line_items[1].unit_price == 150.0
        assert result.line_items[1].line_total == 300.0

    def test_extract_stated_total(self, tmp_path: Path):
        """Test that stated total is extracted from document."""
        pdf_path = tmp_path / "estimate.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 120 >>\nstream\nBT /F1 12 Tf 100 700 Td (Total Estimate: $50000.00) Tj ET\nBT /F1 12 Tf 100 680 Td (front bumper - replace - $50000.00) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000218 00000 n \n"
            b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n298\n%%EOF\n"
        )

        result = extract_repair_tables(pdf_path)

        assert result.stated_total == 50000.0

    def test_calculated_subtotal(self, tmp_path: Path):
        """Test that calculated subtotal matches sum of line items."""
        pdf_path = tmp_path / "estimate.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 120 >>\nstream\nBT /F1 12 Tf 100 700 Td (front bumper - replace - $35000.00) Tj ET\nBT /F1 12 Tf 100 680 Td (paint - paint - $200.00) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000218 00000 n \n"
            b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n298\n%%EOF\n"
        )

        result = extract_repair_tables(pdf_path)

        assert result.calculated_subtotal == 35200.0


class TestTextLineItemParsing:
    """Tests for the text-based line item parsing functions."""

    def test_is_header_line(self):
        """Test header line detection."""
        assert _is_header_line("Part | Qty | Unit Price | Total") is True
        assert _is_header_line("Item Description  Qty  Unit Price  Total") is True
        assert _is_header_line("PART  QUANTITY  UNIT PRICE  TOTAL") is True
        assert _is_header_line("front bumper - replace - $50000.00") is False
        assert _is_header_line("Part  Qty  $50000.00") is False  # Has cost

    def test_parse_simple_line(self):
        """Test parsing simple 'Part - Op - $Cost' format."""
        import re
        match = _SIMPLE_LINE_ITEM_PATTERN.match("front bumper - replace - $50000.00")
        assert match is not None
        item = _parse_simple_line(match)
        assert item is not None
        assert item.part_name == "front bumper"
        assert item.operation == "replace"
        assert item.line_total == 50000.0
        assert item.quantity == 1.0
        assert item.unit_price == 50000.0

    def test_parse_simple_line_no_op(self):
        """Test parsing 'Part - $Cost' format without operation."""
        import re
        match = _SIMPLE_LINE_ITEM_PATTERN.match("rear quarter panel - $400.00")
        assert match is not None
        item = _parse_simple_line(match)
        assert item is not None
        assert item.part_name == "rear quarter panel"
        assert item.operation is None
        assert item.line_total == 400.0

    def test_parse_qty_unit_line(self):
        """Test parsing 'Part Qty UnitPrice Total' format."""
        import re
        match = _QTY_UNIT_TOTAL_PATTERN.match("Front Bumper  1  35000.00  35000.00")
        assert match is not None
        item = _parse_qty_unit_line(match)
        assert item is not None
        assert item.part_name == "Front Bumper"
        assert item.quantity == 1.0
        assert item.unit_price == 35000.0
        assert item.line_total == 35000.0

    def test_parse_qty_unit_line_derives_total(self):
        """Test that total is derived from qty * unit_price when missing."""
        import re
        match = _QTY_UNIT_TOTAL_PATTERN.match("Paint  2  150.00  300.00")
        assert match is not None
        item = _parse_qty_unit_line(match)
        assert item is not None
        assert item.part_name == "Paint"
        assert item.quantity == 2.0
        assert item.unit_price == 150.0
        assert item.line_total == 300.0

    def test_extract_line_items_from_text(self):
        """Test extracting multiple line items from text."""
        text = """
        REPAIR ESTIMATE
        front bumper - replace - $50000.00
        rear quarter panel - repair - $400.00
        paint - paint - $200.00
        """
        items = _extract_line_items_from_text(text)

        assert len(items) == 3
        assert items[0].part_name == "front bumper"
        assert items[0].operation == "replace"
        assert items[0].line_total == 50000.0
        assert items[1].part_name == "rear quarter panel"
        assert items[1].operation == "repair"
        assert items[1].line_total == 400.0
        assert items[2].part_name == "paint"
        assert items[2].operation == "paint"
        assert items[2].line_total == 200.0

    def test_extract_line_items_skips_headers(self):
        """Test that header lines are skipped."""
        text = """
        Part | Qty | Unit Price | Total
        front bumper - replace - $50000.00
        paint - paint - $200.00
        """
        items = _extract_line_items_from_text(text)

        assert len(items) == 2
        assert items[0].part_name == "front bumper"
        assert items[1].part_name == "paint"


class TestTableExtractionIntegration:
    """Integration tests for table extraction in the document pipeline."""

    def test_table_extraction_stored_in_extracted_fields(self, tmp_path: Path):
        """Test that table extraction results are stored in extracted_fields."""
        from app.services.document_extractor import DocumentExtractor
        from app.models.enums import DocType

        pdf_path = tmp_path / "estimate.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 100 >>\nstream\nBT /F1 12 Tf 100 700 Td (REPAIR ESTIMATE) Tj ET\nBT /F1 12 Tf 100 680 Td (front bumper - replace - $50000.00) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000218 00000 n \n"
            b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n298\n%%EOF\n"
        )

        extractor = DocumentExtractor()
        result = extractor.extract(pdf_path, DocType.estimate)

        assert result.extraction_method == "text"
        assert "repair_items" in result.fields
        assert len(result.fields["repair_items"]) == 1
        assert result.fields["repair_items"][0]["part_name"] == "front bumper"
        assert result.fields["repair_items"][0]["operation"] == "replace"
        assert result.fields["repair_items"][0]["line_total"] == 50000.0
        assert result.fields["table_extraction"]["success"] is True
        assert result.fields["table_extraction"]["item_count"] == 1

    def test_table_extraction_adds_structured_fields(self, tmp_path: Path):
        """Test that table extraction adds structured fields to extracted_fields."""
        from app.services.document_extractor import DocumentExtractor
        from app.models.enums import DocType

        pdf_path = tmp_path / "estimate.pdf"
        pdf_path.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
            b"4 0 obj\n<< /Length 120 >>\nstream\nBT /F1 12 Tf 100 700 Td (Front Bumper  1  35000.00  35000.00) Tj ET\nBT /F1 12 Tf 100 680 Td (Paint  2  150.00  300.00) Tj ET\nendstream\nendobj\n"
            b"xref\n0 5\n0000000000 65535 f \n"
            b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000218 00000 n \n"
            b"trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n298\n%%EOF\n"
        )

        extractor = DocumentExtractor()
        result = extractor.extract(pdf_path, DocType.estimate)

        assert result.fields["repair_items"][0]["quantity"] == 1.0
        assert result.fields["repair_items"][0]["unit_price"] == 35000.0
        assert result.fields["repair_items"][1]["quantity"] == 2.0
        assert result.fields["repair_items"][1]["unit_price"] == 150.0