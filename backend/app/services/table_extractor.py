"""Table extraction for repair estimates and invoices using pdfplumber.

This module provides structured table extraction for repair estimate and
invoice documents. It uses pdfplumber to detect and extract tables from
native PDFs, then maps the table columns to structured repair line items.

For OCR-processed documents (scanned PDFs), table extraction is not
available since OCR text loses positional/table structure. The fallback
line-based extraction is used in those cases.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepairLineItem:
    """Structured repair estimate/invoice line item.

    Attributes:
        description: Original line description from the document.
        part_name: Name of the part/component.
        operation: Operation type (e.g., "replace", "repair", "paint").
        quantity: Quantity of the part.
        unit_price: Price per unit.
        labor_hours: Labor hours for this line.
        labor_cost: Labor cost for this line.
        line_total: Total cost for this line (parts + labor).
        raw_text: Original text from the table cell/row for traceability.
    """
    description: str | None = None
    part_name: str | None = None
    operation: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    labor_hours: float | None = None
    labor_cost: float | None = None
    line_total: float | None = None
    raw_text: str | None = None


@dataclass(frozen=True)
class TableExtractionResult:
    """Result of table extraction.

    Attributes:
        line_items: List of structured repair line items.
        stated_total: The document's stated grand total (if found).
        calculated_subtotal: Sum of line item totals.
        warnings: Non-fatal issues encountered.
        success: True if at least one table was extracted.
    """
    line_items: tuple[RepairLineItem, ...]
    stated_total: float | None
    calculated_subtotal: float | None
    warnings: tuple[str, ...]
    success: bool


# Column name patterns for identifying table columns
_PART_COL_PATTERNS = [
    r"part", r"item", r"description", r"component", r"part\s*#", r"part\s*number",
]
_QTY_COL_PATTERNS = [
    r"qty", r"quantity", r"q\s*\.?\s*ty", r"units?", r"count",
]
_UNIT_PRICE_COL_PATTERNS = [
    r"unit\s*price", r"price\s*per", r"rate", r"unit\s*cost", r"cost\s*per",
]
_LABOR_HOURS_COL_PATTERNS = [
    r"labor\s*hours?", r"hrs?", r"hours?", r"labor\s*time",
]
_LABOR_COST_COL_PATTERNS = [
    r"labor\s*cost", r"labor\s*\$", r"labour\s*cost",
]
_LINE_TOTAL_COL_PATTERNS = [
    r"line\s*total", r"total", r"amount", r"extended\s*price", r"ext\s*price",
]
_OPERATION_COL_PATTERNS = [
    r"operation", r"op\s*\.?", r"type", r"work", r"service",
]

# Total amount patterns for finding stated document total
_TOTAL_PATTERNS = [
    r"(?:grand\s*total|total\s*amount|amount\s*due|total\s*due|balance\s*due)[:\s]*[$€£]?\s*([\d,]+\.?\d*)",
    r"(?:total\s*(?:estimate|amount)?)[:\s]*[$€£]?\s*([\d,]+\.?\d*)",
]

# Operation type normalization
_OPERATION_MAP = {
    "replace": "replace",
    "repl": "replace",
    "repair": "repair",
    "rpr": "repair",
    "paint": "paint",
    "pnt": "paint",
    "refinish": "refinish",
    "r&i": "remove_and_install",
    "remove and install": "remove_and_install",
    "remove": "remove",
    "install": "install",
    "align": "align",
    "alignment": "align",
    "diagnose": "diagnose",
    "diag": "diagnose",
    "inspect": "inspect",
    "section": "section",
    "sublet": "sublet",
}


def _normalize_column_name(name: str) -> str:
    """Normalize column name for matching."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _match_column(name: str, patterns: list[str]) -> bool:
    """Check if column name matches any pattern."""
    normalized = _normalize_column_name(name)
    for pattern in patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            return True
    return False


def _identify_columns(headers: list[str]) -> dict[str, int]:
    """Identify table column indices by header names.

    Returns a dict mapping semantic column names to column indices.
    """
    result = {}
    for i, header in enumerate(headers):
        if not header:
            continue
        if _match_column(header, _PART_COL_PATTERNS):
            result["part"] = i
        elif _match_column(header, _QTY_COL_PATTERNS):
            result["qty"] = i
        elif _match_column(header, _UNIT_PRICE_COL_PATTERNS):
            result["unit_price"] = i
        elif _match_column(header, _LABOR_HOURS_COL_PATTERNS):
            result["labor_hours"] = i
        elif _match_column(header, _LABOR_COST_COL_PATTERNS):
            result["labor_cost"] = i
        elif _match_column(header, _LINE_TOTAL_COL_PATTERNS):
            result["line_total"] = i
        elif _match_column(header, _OPERATION_COL_PATTERNS):
            result["operation"] = i
    return result


def _parse_float(value: str | None) -> float | None:
    """Parse a float from a string, handling currency and commas."""
    if not value:
        return None
    # Remove currency symbols, commas, and whitespace
    cleaned = re.sub(r"[$€£,\s]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    """Parse an int from a string."""
    if not value:
        return None
    cleaned = re.sub(r"[,\s]", "", value)
    try:
        return int(float(cleaned))  # Handle "1.0" -> 1
    except ValueError:
        return None


def _normalize_operation(value: str | None) -> str | None:
    """Normalize operation string to standard values."""
    if not value:
        return None
    normalized = value.lower().strip()
    return _OPERATION_MAP.get(normalized, value)


def _extract_stated_total(text: str) -> float | None:
    """Extract the stated document total from text."""
    for pattern in _TOTAL_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _parse_float(match.group(1))
    return None


def extract_repair_tables(file_path: Path) -> TableExtractionResult:
    """Extract repair estimate/invoice tables from a PDF.

    Uses pdfplumber to find and extract tables from the document.
    Attempts to map table columns to structured repair line items.

    Args:
        file_path: Path to the PDF file.

    Returns:
        TableExtractionResult with extracted line items and metadata.
    """
    warnings: list[str] = []
    all_items: list[RepairLineItem] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            full_text = ""
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"

                # Extract tables from this page
                tables = page.extract_tables()
                if not tables:
                    continue

                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue  # Need at least header + 1 data row

                    # First row is typically headers
                    headers = [str(cell or "").strip() for cell in table[0]]
                    col_map = _identify_columns(headers)

                    if "part" not in col_map and "line_total" not in col_map:
                        # Doesn't look like a repair estimate table
                        continue

                    # Process data rows
                    for row_idx, row in enumerate(table[1:], start=1):
                        if not row or all(not cell for cell in row):
                            continue

                        # Extract values from mapped columns
                        part_name = None
                        operation = None
                        quantity = None
                        unit_price = None
                        labor_hours = None
                        labor_cost = None
                        line_total = None
                        raw_parts = []

                        for col_name, col_idx in col_map.items():
                            if col_idx < len(row):
                                cell = row[col_idx]
                                if cell:
                                    raw_parts.append(f"{col_name}: {cell}")

                        if not raw_parts:
                            continue

                        # Parse each field
                        if "part" in col_map and col_map["part"] < len(row):
                            part_name = str(row[col_map["part"]] or "").strip() or None
                        if "operation" in col_map and col_map["operation"] < len(row):
                            operation = _normalize_operation(str(row[col_map["operation"]] or "").strip() or None)
                        if "qty" in col_map and col_map["qty"] < len(row):
                            quantity = _parse_float(str(row[col_map["qty"]] or ""))
                        if "unit_price" in col_map and col_map["unit_price"] < len(row):
                            unit_price = _parse_float(str(row[col_map["unit_price"]] or ""))
                        if "labor_hours" in col_map and col_map["labor_hours"] < len(row):
                            labor_hours = _parse_float(str(row[col_map["labor_hours"]] or ""))
                        if "labor_cost" in col_map and col_map["labor_cost"] < len(row):
                            labor_cost = _parse_float(str(row[col_map["labor_cost"]] or ""))
                        if "line_total" in col_map and col_map["line_total"] < len(row):
                            line_total = _parse_float(str(row[col_map["line_total"]] or ""))

                        # Try to derive missing line_total from quantity * unit_price + labor_cost
                        if line_total is None:
                            parts_cost = (quantity or 0) * (unit_price or 0)
                            if parts_cost > 0 or labor_cost is not None:
                                line_total = parts_cost + (labor_cost or 0)

                        # Build description from available fields
                        desc_parts = []
                        if part_name:
                            desc_parts.append(part_name)
                        if operation:
                            desc_parts.append(f"({operation})")
                        if quantity and quantity != 1:
                            desc_parts.append(f"Qty: {quantity}")
                        description = " ".join(desc_parts) if desc_parts else None

                        raw_text = "; ".join(raw_parts) if raw_parts else None

                        item = RepairLineItem(
                            description=description,
                            part_name=part_name,
                            operation=operation,
                            quantity=quantity,
                            unit_price=unit_price,
                            labor_hours=labor_hours,
                            labor_cost=labor_cost,
                            line_total=line_total,
                            raw_text=raw_text,
                        )
                        all_items.append(item)

    except Exception as e:
        logger.exception("Table extraction failed for %s: %s", file_path, e)
        warnings.append(f"Table extraction error: {e}")
        return TableExtractionResult(
            line_items=(),
            stated_total=None,
            calculated_subtotal=None,
            warnings=tuple(warnings),
            success=False,
        )

    # Extract stated total from full text
    stated_total = _extract_stated_total(full_text)

    # If no items from tables, try text-based extraction as fallback
    if not all_items:
        text_items = _extract_line_items_from_text(full_text)
        all_items.extend(text_items)

    # Calculate subtotal from line items
    calculated_subtotal = None
    items_with_totals = [item for item in all_items if item.line_total is not None]
    if items_with_totals:
        calculated_subtotal = sum(item.line_total for item in items_with_totals)

    return TableExtractionResult(
        line_items=tuple(all_items),
        stated_total=stated_total,
        calculated_subtotal=calculated_subtotal,
        warnings=tuple(warnings),
        success=len(all_items) > 0,
    )


__all__ = [
    "RepairLineItem",
    "TableExtractionResult",
    "extract_repair_tables",
]


# ─── Text-based line item extraction (fallback for non-table PDFs) ──────────


# Pattern to match lines like:
# "front bumper - replace - $80000.00"
# "front bumper replace 1 35000.00 2.5 5000.00 40000.00"
# "Part Name | Qty | Unit Price | Labor Hrs | Labor Cost | Total"
_TEXT_LINE_ITEM_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<desc>.+?)                    # Description/part name (greedy)
    (?:\s*[-|]\s*|\s{2,})            # Separator: dash, pipe, or multiple spaces
    (?P<op>replace|repair|paint|refinish|r&i|remove|install|align|diagnose|inspect|section|sublet|rpr|repl|pnt|r&i)?\s*   # Optional operation
    (?:[-|]\s*|\s{2,})?              # Optional separator
    (?P<qty>\d+(?:\.\d+)?)?          # Optional quantity
    \s*[-|]?\s*
    (?P<unit_price>[\d,]+\.?\d*)?    # Optional unit price
    \s*[-|]?\s*
    (?P<labor_hrs>\d+(?:\.\d+)?)?    # Optional labor hours
    \s*[-|]?\s*
    (?P<labor_cost>[\d,]+\.?\d*)?    # Optional labor cost
    \s*[-|]?\s*
    (?P<line_total>[\d,]+\.?\d*)     # Line total (required)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Simpler pattern for lines like "front bumper - replace - $80000.00"
# Also matches "part name - $cost" without operation
_SIMPLE_LINE_ITEM_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<part>[^$]+?)                # Part name/description
    \s*[-|]\s*
    (?:
        (?P<op>replace|repair|paint|refinish|r&i|remove|install|align|diagnose|inspect|section|sublet|rpr|repl|pnt|r&i)\s*
        \s*[-|]\s*
    )?
    (?P<cost>[$€£]?\s*[\d,]+\.\d{2})  # Cost at end
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pattern for lines with quantity and unit price: "Part Name  Qty  UnitPrice  Total"
_QTY_UNIT_TOTAL_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<part>.+?)
    \s+
    (?P<qty>\d+(?:\.\d+)?)
    \s+
    (?P<unit_price>[$€£]?\s*[\d,]+\.\d{2})
    \s+
    (?P<line_total>[$€£]?\s*[\d,]+\.\d{2})
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _extract_line_items_from_text(text: str) -> list[RepairLineItem]:
    """Extract structured line items from plain text that looks like a table.

    This is a fallback for PDFs that don't have actual table structures
    but contain text that looks like a table (e.g., lines with parts and costs).

    Args:
        text: The full text content of the document.

    Returns:
        List of structured RepairLineItem objects.
    """
    items: list[RepairLineItem] = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue

        # Skip header-like lines
        if _is_header_line(line):
            continue

        # Try patterns in order of specificity
        item = None

        # Pattern 1: Complex structured line with qty, unit_price, labor, total
        match = _TEXT_LINE_ITEM_PATTERN.match(line)
        if match:
            item = _parse_complex_line(match)
        else:
            # Pattern 2: Simple "Part - Op - $Cost" format
            match = _SIMPLE_LINE_ITEM_PATTERN.match(line)
            if match:
                item = _parse_simple_line(match)
            else:
                # Pattern 3: Qty UnitPrice Total format
                match = _QTY_UNIT_TOTAL_PATTERN.match(line)
                if match:
                    item = _parse_qty_unit_line(match)

        if item:
            items.append(item)

    return items


def _is_header_line(line: str) -> bool:
    """Check if a line is a table header."""
    line_lower = line.lower()
    header_keywords = [
        "part", "item", "description", "component",
        "qty", "quantity",
        "unit price", "unit cost", "rate",
        "labor", "hrs", "hours",
        "total", "amount", "extended",
        "operation", "op",
    ]
    # Count how many header keywords appear
    matches = sum(1 for kw in header_keywords if kw in line_lower)
    # If it has multiple header keywords and no dollar amounts, it's likely a header
    has_cost = bool(re.search(r"[$€£]", line))
    return matches >= 2 and not has_cost


def _parse_complex_line(match: re.Match) -> RepairLineItem | None:
    """Parse a line matched by the complex pattern."""
    try:
        desc = match.group("desc").strip() if match.group("desc") else None
        op = _normalize_operation(match.group("op")) if match.group("op") else None
        qty = _parse_float(match.group("qty")) if match.group("qty") else None
        unit_price = _parse_float(match.group("unit_price")) if match.group("unit_price") else None
        labor_hrs = _parse_float(match.group("labor_hrs")) if match.group("labor_hrs") else None
        labor_cost = _parse_float(match.group("labor_cost")) if match.group("labor_cost") else None
        line_total = _parse_float(match.group("line_total")) if match.group("line_total") else None

        # Try to extract part_name from description
        part_name = None
        if desc:
            # Remove operation from description if present
            if op and op in desc.lower():
                part_name = desc.replace(op, "").strip(" -|")
            else:
                part_name = desc

        return RepairLineItem(
            description=desc,
            part_name=part_name,
            operation=op,
            quantity=qty,
            unit_price=unit_price,
            labor_hours=labor_hrs,
            labor_cost=labor_cost,
            line_total=line_total,
            raw_text=match.group(0),
        )
    except Exception:
        return None


def _parse_simple_line(match: re.Match) -> RepairLineItem | None:
    """Parse a line matched by the simple pattern (Part - Op - $Cost)."""
    try:
        part_name = match.group("part").strip() if match.group("part") else None
        op = _normalize_operation(match.group("op")) if match.group("op") else None
        cost = _parse_float(match.group("cost"))
        if cost is None:
            return None

        return RepairLineItem(
            description=part_name,
            part_name=part_name,
            operation=op,
            quantity=1.0,
            unit_price=cost,
            line_total=cost,
            raw_text=match.group(0),
        )
    except Exception:
        return None


def _parse_qty_unit_line(match: re.Match) -> RepairLineItem | None:
    """Parse a line with Qty UnitPrice Total format."""
    try:
        part_name = match.group("part").strip() if match.group("part") else None
        qty = _parse_float(match.group("qty"))
        unit_price = _parse_float(match.group("unit_price"))
        line_total = _parse_float(match.group("line_total"))

        if line_total is None and qty and unit_price:
            line_total = qty * unit_price

        return RepairLineItem(
            description=part_name,
            part_name=part_name,
            quantity=qty,
            unit_price=unit_price,
            line_total=line_total,
            raw_text=match.group(0),
        )
    except Exception:
        return None