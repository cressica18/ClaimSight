"""OCR Adapter — provides a clean interface for OCR operations.

This module isolates the OCR implementation (pytesseract + Tesseract)
behind a simple interface so the rest of ClaimSight does not depend
directly on a specific OCR library.

For scanned/image-only PDFs, the DocumentExtractor will:
1. Render PDF pages to images using PyMuPDF
2. Call the OCR adapter to extract text from each page image
3. Combine the page text and feed it into the existing field extraction logic

The OCR adapter is optional — if Tesseract is not available, the system
gracefully falls back to the native extraction result.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRResult:
    """Result of OCR processing.

    Attributes:
        text: Combined text from all pages.
        page_count: Number of pages processed.
        warnings: Non-fatal issues encountered during OCR.
        success: True if OCR completed (even if text is minimal).
    """
    text: str
    page_count: int
    warnings: tuple[str, ...]
    success: bool


class OCRAdapter:
    """Adapter for Tesseract OCR via pytesseract.

    Renders PDF pages to images using PyMuPDF, then runs Tesseract OCR.
    Designed to be a drop-in replacement for native text extraction
    when the PDF appears to be scanned/image-only.
    """

    # Maximum pages to process (prevent resource exhaustion)
    MAX_PAGES = 50

    # Maximum image dimension for OCR (prevent memory issues)
    MAX_IMAGE_DIMENSION = 3000

    # DPI for rendering PDF pages to images
    RENDER_DPI = 300

    def __init__(self) -> None:
        self._warnings: list[str] = []
        self._tesseract_available = self._check_tesseract()

    def _check_tesseract(self) -> bool:
        """Check if Tesseract binary is available."""
        return shutil.which("tesseract") is not None

    def is_available(self) -> bool:
        """Return True if OCR can be performed."""
        return self._tesseract_available

    def extract_text_from_pdf(self, file_path: Path) -> OCRResult:
        """Extract text from a PDF using OCR.

        Args:
            file_path: Path to the PDF file.

        Returns:
            OCRResult with combined text from all pages.
        """
        self._warnings = []

        if not self._tesseract_available:
            self._warnings.append("Tesseract not installed; OCR unavailable.")
            return OCRResult(
                text="",
                page_count=0,
                warnings=tuple(self._warnings),
                success=False,
            )

        if not file_path.exists():
            self._warnings.append(f"File not found: {file_path}")
            return OCRResult(
                text="",
                page_count=0,
                warnings=tuple(self._warnings),
                success=False,
            )

        try:
            doc = fitz.open(file_path)
            page_count = doc.page_count

            if page_count == 0:
                self._warnings.append("PDF has zero pages.")
                doc.close()
                return OCRResult(
                    text="",
                    page_count=0,
                    warnings=tuple(self._warnings),
                    success=False,
                )

            if page_count > self.MAX_PAGES:
                self._warnings.append(
                    f"PDF has {page_count} pages; limiting OCR to first {self.MAX_PAGES}."
                )
                page_count = self.MAX_PAGES

            text_parts = []
            for page_num in range(page_count):
                page_text = self._ocr_page(doc, page_num)
                if page_text:
                    text_parts.append(page_text)

            doc.close()

            combined_text = "\n\n--- PAGE BREAK ---\n\n".join(text_parts)

            return OCRResult(
                text=combined_text,
                page_count=page_count,
                warnings=tuple(self._warnings),
                success=True,
            )

        except Exception as e:
            logger.exception("OCR failed for %s: %s", file_path, e)
            self._warnings.append(f"OCR error: {e}")
            return OCRResult(
                text="",
                page_count=0,
                warnings=tuple(self._warnings),
                success=False,
            )

    def _ocr_page(self, doc: fitz.Document, page_num: int) -> str:
        """OCR a single PDF page.

        Renders the page to a high-DPI image, then runs Tesseract.

        Args:
            doc: Open PyMuPDF document.
            page_num: Page number (0-indexed).

        Returns:
            Extracted text from the page, or empty string on failure.
        """
        try:
            page = doc[page_num]

            # Calculate zoom to achieve target DPI
            zoom = self.RENDER_DPI / 72.0
            mat = fitz.Matrix(zoom, zoom)

            # Render page to pixmap (image)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Check image dimensions
            if pix.width > self.MAX_IMAGE_DIMENSION or pix.height > self.MAX_IMAGE_DIMENSION:
                # Downscale if too large
                scale = min(
                    self.MAX_IMAGE_DIMENSION / pix.width,
                    self.MAX_IMAGE_DIMENSION / pix.height,
                )
                new_zoom = zoom * scale
                mat = fitz.Matrix(new_zoom, new_zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)

            # Convert to PIL Image for pytesseract
            # PyMuPDF pixmap can be converted to bytes then PIL Image
            img_bytes = pix.tobytes("png")

            # Import pytesseract lazily to avoid hard dependency at import time
            import pytesseract
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(img_bytes))

            # Run OCR
            # Use default config; could add --psm 6 for uniform text blocks
            text = pytesseract.image_to_string(img)

            return text.strip()

        except Exception as e:
            logger.warning("OCR failed on page %d: %s", page_num, e)
            self._warnings.append(f"Page {page_num + 1} OCR failed: {e}")
            return ""


def create_ocr_adapter() -> OCRAdapter:
    """Factory function to create an OCR adapter instance.

    Allows tests to inject a mock adapter if needed.
    """
    return OCRAdapter()


__all__ = [
    "OCRAdapter",
    "OCRResult",
    "create_ocr_adapter",
]