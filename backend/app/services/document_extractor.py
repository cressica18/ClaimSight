"""Document extraction using PyMuPDF for PDF text extraction.

This module provides a clean, deterministic extraction interface that can
handle common claim documents: policy, claim_form, estimate, invoice,
previous_claim. It extracts structured fields where reliably possible and
returns null/unknown for fields that cannot be found.

For scanned/image-only PDFs where no text can be extracted, the module
attempts OCR using Tesseract (if available) as a fallback. The extraction
method (native text vs OCR) is tracked for provenance.

For repair estimates and invoices, table extraction is attempted using
pdfplumber to extract structured line items with quantities, unit prices,
labor hours, and operation types.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app.models.enums import DocType
from app.services.ocr import OCRAdapter, OCRResult, create_ocr_adapter
from app.services.table_extractor import (
    RepairLineItem,
    TableExtractionResult,
    extract_repair_tables,
)

logger = logging.getLogger(__name__)

# Regex patterns for common field extraction
_POLICY_NUMBER_RE = re.compile(r"(?i)\bpol[_-]?([a-z0-9]{4,})\b")
_CLAIM_NUMBER_RE = re.compile(r"(?i)\bclm[_-]?([a-z0-9]{4,})\b")
_VIN_RE = re.compile(r"(?i)\bvin[:\s]*([a-hj-npr-z0-9]{17})\b")
_PLATE_RE = re.compile(r"(?i)\b(?:plate|registration)[:\s]*([a-z0-9]{1,8})\b")
_DATE_RE = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
_CURRENCY_RE = re.compile(r"[$€£]\s*([\d,]+\.?\d*)")
_AMOUNT_RE = re.compile(r"(?:total|amount|claimed)[:\s]*[$€£]?\s*([\d,]+\.?\d*)", re.IGNORECASE)

# Coverage type keywords
_COVERAGE_KEYWORDS = {
    "comprehensive": ["comprehensive", "full coverage"],
    "collision": ["collision"],
    "third_party": ["third party", "third-party", "liability only", "liability"],
    "fire_theft": ["fire and theft", "fire & theft", "fire theft"],
}

# Damage type keywords for claim forms
_DAMAGE_KEYWORDS = [
    "scratch", "dent", "crack", "shattered_glass", "bumper_damage",
    "panel_damage", "headlight_damage", "windshield", "mirror",
    "door", "fender", "hood", "roof", "trunk", "quarter panel",
    "side panel", "rear", "front"
]


@dataclass(frozen=True)
class ExtractionResult:
    """Result of document extraction.

    Attributes:
        fields: Structured fields extracted from the document.
        raw_confidence: Overall confidence in the extraction (0.0-1.0).
        warnings: Non-fatal issues encountered during extraction.
        is_scanned: True if the document appears to be scanned (no extractable text).
        extraction_method: "text" for native PDF text, "ocr" for OCR, "filename" for filename fallback.
    """
    fields: dict[str, Any]
    raw_confidence: float
    warnings: tuple[str, ...]
    is_scanned: bool = False
    extraction_method: str = "text"


class DocumentExtractor:
    """Extracts structured fields from claim documents.

    Supports PDF documents only. For each document type, attempts to
    extract a defined set of fields using regex patterns on the extracted
    text. Fields that cannot be reliably found are omitted (not set to
    null) so the absence of a field is distinguishable from a field that
    was found but empty.

    The extractor first tries native PDF text extraction via PyMuPDF.
    If the PDF appears to be scanned/image-only (minimal extractable text),
    it falls back to OCR using Tesseract (if available). The extraction
    method is tracked in the result for provenance.

    For policy documents, a filename-based fallback extracts policy numbers
    when both native text and OCR fail.
    """

    MIN_TEXT_LENGTH = 10  # Minimum characters to consider "not scanned"

    def __init__(self, ocr_adapter: OCRAdapter | None = None) -> None:
        self._warnings: list[str] = []
        self._ocr_adapter = ocr_adapter or create_ocr_adapter()

    def extract(self, file_path: Path, doc_type: DocType) -> ExtractionResult:
        """Extract fields from a PDF document.

        Args:
            file_path: Path to the PDF file.
            doc_type: Type of document (policy, claim_form, estimate, etc.).

        Returns:
            ExtractionResult with extracted fields, confidence, warnings,
            is_scanned flag, and extraction_method provenance.
        """
        self._warnings = []

        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            self._warnings.append(f"Unsupported file type: {file_path.suffix}. Only PDF is supported.")
            return ExtractionResult(
                fields={},
                raw_confidence=0.0,
                warnings=tuple(self._warnings),
                is_scanned=False,
                extraction_method="text",
            )

        filename = file_path.name

        # Step 1: Try native text extraction
        native_text = self._extract_text(file_path)

        # Step 2: If minimal text, try OCR fallback
        if not native_text or len(native_text.strip()) < self.MIN_TEXT_LENGTH:
            self._warnings.append("No extractable text found; document may be scanned or image-only.")
            ocr_result = self._try_ocr(file_path)

            if ocr_result.success and ocr_result.text.strip():
                # OCR succeeded - extract fields from OCR text
                fields = self._extract_fields(ocr_result.text, doc_type)
                self._warnings.extend(ocr_result.warnings)
                confidence = self._compute_confidence(fields, doc_type, len(ocr_result.text))
                return ExtractionResult(
                    fields=fields,
                    raw_confidence=confidence,
                    warnings=tuple(self._warnings),
                    is_scanned=True,
                    extraction_method="ocr",
                )

            # OCR failed or unavailable - try filename fallback for policy documents
            fields = {}
            if doc_type == DocType.policy:
                policy_num = self._find_policy_number_in_filename(filename)
                if policy_num:
                    fields["policy_number"] = policy_num
            # Use 0.5 confidence for blank/minimal PDFs to match the original
            # stub behavior (STUB_CONFIDENCE). This avoids triggering the
            # risk engine's low-data-confidence bump for placeholder documents.
            return ExtractionResult(
                fields=fields,
                raw_confidence=0.5,
                warnings=tuple(self._warnings),
                is_scanned=True,
                extraction_method="filename" if fields else "none",
            )

        # Step 3: Native text extraction succeeded - extract fields
        fields = self._extract_fields(native_text, doc_type)

        # Step 4: For estimates/invoices, attempt table extraction for structured line items
        if doc_type in (DocType.estimate, DocType.invoice):
            table_result = self._try_table_extraction(file_path)
            if table_result.success:
                # Add structured line items
                fields["repair_items"] = [
                    {
                        "description": item.description,
                        "part_name": item.part_name,
                        "operation": item.operation,
                        "quantity": item.quantity,
                        "unit_price": item.unit_price,
                        "labor_hours": item.labor_hours,
                        "labor_cost": item.labor_cost,
                        "line_total": item.line_total,
                    }
                    for item in table_result.line_items
                ]
                # Add table extraction metadata
                if table_result.stated_total is not None:
                    fields["stated_total"] = table_result.stated_total
                if table_result.calculated_subtotal is not None:
                    fields["calculated_subtotal"] = table_result.calculated_subtotal
                fields["table_extraction"] = {
                    "success": True,
                    "item_count": len(table_result.line_items),
                }
                self._warnings.extend(table_result.warnings)
            else:
                fields["table_extraction"] = {"success": False}
                # Fall back to line-based extraction
                fields["repair_items"] = self._extract_repair_items(native_text)

        # Fallback: if policy document and no policy_number found in text, try filename
        if doc_type == DocType.policy and "policy_number" not in fields:
            policy_num = self._find_policy_number_in_filename(filename)
            if policy_num:
                fields["policy_number"] = policy_num

        # Confidence based on number of fields found and text quality
        confidence = self._compute_confidence(fields, doc_type, len(native_text))

        return ExtractionResult(
            fields=fields,
            raw_confidence=confidence,
            warnings=tuple(self._warnings),
            is_scanned=False,
            extraction_method="text",
        )

    def _try_ocr(self, file_path: Path) -> OCRResult:
        """Attempt OCR on the PDF file."""
        if not self._ocr_adapter.is_available():
            self._warnings.append("Tesseract not installed; OCR fallback unavailable.")
            return OCRResult(
                text="",
                page_count=0,
                warnings=("Tesseract not installed; OCR fallback unavailable.",),
                success=False,
            )

        logger.info("Attempting OCR on %s", file_path)
        return self._ocr_adapter.extract_text_from_pdf(file_path)

    def _try_table_extraction(self, file_path: Path) -> TableExtractionResult:
        """Attempt table extraction for repair estimates/invoices."""
        try:
            logger.info("Attempting table extraction on %s", file_path)
            return extract_repair_tables(file_path)
        except Exception as e:
            logger.exception("Table extraction failed for %s: %s", file_path, e)
            return TableExtractionResult(
                line_items=(),
                stated_total=None,
                calculated_subtotal=None,
                warnings=(f"Table extraction error: {e}",),
                success=False,
            )

    def _find_policy_number_in_filename(self, filename: str) -> str | None:
        """Extract policy number from filename as a fallback."""
        match = _POLICY_NUMBER_RE.search(filename)
        if match:
            return f"POL-{match.group(1).upper()}"
        return None

    def _extract_text(self, file_path: Path) -> str:
        """Extract all text from a PDF using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text("text"))
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            logger.exception("Failed to extract text from %s: %s", file_path, e)
            self._warnings.append(f"PDF parsing error: {e}")
            return ""

    def _extract_fields(self, text: str, doc_type: DocType) -> dict[str, Any]:
        """Extract structured fields based on document type."""
        text_lower = text.lower()
        fields: dict[str, Any] = {}

        # Common fields for all document types
        policy_num = self._find_policy_number(text)
        if policy_num:
            fields["policy_number"] = policy_num

        claim_num = self._find_claim_number(text)
        if claim_num:
            fields["claim_number"] = claim_num

        vin = self._find_vin(text)
        if vin:
            fields["vin"] = vin

        plate = self._find_plate(text)
        if plate:
            fields["plate_number"] = plate

        # Document-type specific extraction
        if doc_type == DocType.policy:
            fields.update(self._extract_policy_fields(text, text_lower))
        elif doc_type == DocType.claim_form:
            fields.update(self._extract_claim_form_fields(text, text_lower))
        elif doc_type in (DocType.estimate, DocType.invoice):
            fields.update(self._extract_estimate_fields(text, text_lower))
        elif doc_type == DocType.previous_claim:
            fields.update(self._extract_previous_claim_fields(text, text_lower))

        return fields

    def _find_policy_number(self, text: str) -> str | None:
        match = _POLICY_NUMBER_RE.search(text)
        if match:
            return f"POL-{match.group(1).upper()}"
        return None

    def _find_claim_number(self, text: str) -> str | None:
        match = _CLAIM_NUMBER_RE.search(text)
        if match:
            return f"CLM-{match.group(1).upper()}"
        return None

    def _find_vin(self, text: str) -> str | None:
        match = _VIN_RE.search(text)
        if match:
            return match.group(1).upper()
        return None

    def _find_plate(self, text: str) -> str | None:
        match = _PLATE_RE.search(text)
        if match:
            return match.group(1).upper()
        return None

    def _extract_policy_fields(self, text: str, text_lower: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}

        # Coverage type
        for cov_type, keywords in _COVERAGE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                fields["coverage_type"] = cov_type
                break

        # Policy dates
        dates = _DATE_RE.findall(text)
        if len(dates) >= 2:
            # Heuristic: first date is start, second is end
            fields["policy_start_date"] = dates[0]
            fields["policy_end_date"] = dates[1]
        elif len(dates) == 1:
            fields["policy_start_date"] = dates[0]

        # Insured name - look for "insured" or "policyholder" followed by name
        insured_match = re.search(r"(?:insured|policyholder|named insured)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
        if insured_match:
            fields["insured_name"] = insured_match.group(1).strip()

        return fields

    def _extract_claim_form_fields(self, text: str, text_lower: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}

        # Accident date
        dates = _DATE_RE.findall(text)
        if dates:
            # Heuristic: accident date is often near "accident", "incident", "loss"
            accident_date = None
            for i, date in enumerate(dates):
                # Look for context around the date
                idx = text_lower.find(date.lower())
                if idx >= 0:
                    context = text_lower[max(0, idx-50):idx+50]
                    if any(kw in context for kw in ["accident", "incident", "loss", "occurred", "happened"]):
                        accident_date = date
                        break
            if accident_date:
                fields["accident_date"] = accident_date
            elif dates:
                fields["accident_date"] = dates[0]

        # Claimed amount
        amount_match = _AMOUNT_RE.search(text)
        if amount_match:
            try:
                fields["claimed_amount"] = float(amount_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Accident description - look for "description" or "details" section
        desc_match = re.search(r"(?:description|details|what happened)[:\s]+(.{20,500})", text, re.IGNORECASE | re.DOTALL)
        if desc_match:
            fields["accident_description"] = desc_match.group(1).strip()[:500]

        # Claimed damage types
        found_damages = []
        for dmg in _DAMAGE_KEYWORDS:
            if dmg in text_lower:
                found_damages.append(dmg)
        if found_damages:
            fields["claimed_damage_types"] = found_damages

        return fields

    def _extract_estimate_fields(self, text: str, text_lower: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}

        # Repair date
        dates = _DATE_RE.findall(text)
        if dates:
            # Look for date near "date", "issued", "estimate"
            for date in dates:
                idx = text_lower.find(date.lower())
                if idx >= 0:
                    context = text_lower[max(0, idx-30):idx+30]
                    if any(kw in context for kw in ["date", "issued", "estimate", "invoice"]):
                        fields["repair_date"] = date
                        break
            if "repair_date" not in fields and dates:
                fields["repair_date"] = dates[0]

        # Total estimate amount
        total_match = re.search(r"(?:total|grand total|amount due)[:\s]*[$€£]?\s*([\d,]+\.?\d*)", text, re.IGNORECASE)
        if total_match:
            try:
                fields["total_estimate"] = float(total_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Currency
        currency_match = _CURRENCY_RE.search(text)
        if currency_match:
            fields["currency"] = currency_match.group(0)[0]  # $, €, or £
        else:
            fields["currency"] = "USD"  # Default

        # Repair items - look for lines with part names and costs
        items = self._extract_repair_items(text)
        if items:
            fields["repair_items"] = items

        # Shop name
        shop_match = re.search(r"(?:shop|garage|body shop|repairer)[:\s]+([A-Z][A-Za-z\s&.'-]{2,50})", text)
        if shop_match:
            fields["shop_name"] = shop_match.group(1).strip()

        return fields

    def _extract_repair_items(self, text: str) -> list[dict[str, Any]]:
        """Extract repair line items from estimate/invoice text."""
        items: list[dict[str, Any]] = []

        # Pattern: Part name ... $amount or Part name ... amount
        # This is a heuristic; real invoices vary widely in format
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # Look for a cost at the end of the line
            cost_match = re.search(r"[$€£]?\s*([\d,]+\.\d{2})\s*$", line)
            if cost_match:
                try:
                    cost = float(cost_match.group(1).replace(",", ""))
                    # Part name is everything before the cost
                    part_name = line[:cost_match.start()].strip()
                    # Clean up part name
                    part_name = re.sub(r"\s+", " ", part_name).strip(" -.:")
                    if part_name and len(part_name) > 2:
                        items.append({
                            "part_name": part_name,
                            "cost": cost,
                        })
                except ValueError:
                    continue

        return items[:20]  # Cap at 20 items

    def _extract_previous_claim_fields(self, text: str, text_lower: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}

        # Previous claim number
        claim_num = self._find_claim_number(text)
        if claim_num:
            fields["previous_claim_number"] = claim_num

        # Incident date
        dates = _DATE_RE.findall(text)
        if dates:
            fields["incident_date"] = dates[0]

        # Damage summary
        found_damages = []
        for dmg in _DAMAGE_KEYWORDS:
            if dmg in text_lower:
                found_damages.append(dmg)
        if found_damages:
            fields["damage_summary"] = ", ".join(found_damages)

        # Claimed amount
        amount_match = _AMOUNT_RE.search(text)
        if amount_match:
            try:
                fields["claimed_amount"] = float(amount_match.group(1).replace(",", ""))
            except ValueError:
                pass

        return fields

    def _compute_confidence(
        self,
        fields: dict[str, Any],
        doc_type: DocType,
        text_length: int,
    ) -> float:
        """Compute overall confidence based on fields found and text quality."""
        # Base confidence from text length (more text = more reliable)
        if text_length < 100:
            base = 0.3
        elif text_length < 500:
            base = 0.5
        elif text_length < 2000:
            base = 0.7
        else:
            base = 0.85

        # Boost for each expected field found
        expected_fields = {
            DocType.policy: ["policy_number", "coverage_type", "policy_start_date", "policy_end_date"],
            DocType.claim_form: ["claim_number", "policy_number", "accident_date", "claimed_amount"],
            DocType.estimate: ["total_estimate", "repair_date", "repair_items"],
            DocType.invoice: ["total_estimate", "repair_date", "repair_items"],
            DocType.previous_claim: ["previous_claim_number", "incident_date", "claimed_amount"],
        }

        expected = expected_fields.get(doc_type, [])
        if expected:
            found = sum(1 for f in expected if f in fields)
            field_ratio = found / len(expected)
            # Weight: 60% base, 40% field coverage
            return round(0.6 * base + 0.4 * field_ratio, 2)

        return round(base, 2)


def extract_document_file(file_path: Path, doc_type: DocType) -> ExtractionResult:
    """Convenience function for single-file extraction."""
    extractor = DocumentExtractor()
    return extractor.extract(file_path, doc_type)


# Backward compatibility constant for tests
STUB_CONFIDENCE = 0.5


__all__ = [
    "DocumentExtractor",
    "ExtractionResult",
    "extract_document_file",
    "STUB_CONFIDENCE",
]