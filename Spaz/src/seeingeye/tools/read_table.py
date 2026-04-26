"""ReadTable tool — ported from ``src/multi-agent/app/tool/read_table.py``.

TPS-03 (simplified 2026-04-13): semantic equivalence with the original.
Preserves the img2table + TesseractOCR backend, all defaults
(``borderless_tables=True``, ``implicit_rows=True``, ``language='eng'``),
the ``_format_table_as_text`` helper, and the output-string template
verbatim.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

_ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".pdf"}


def _format_table_as_text(df: pd.DataFrame) -> str:
    """Format DataFrame as readable text table. Verbatim port of the
    old ``ReadTable._format_table_as_text`` method."""
    if df.empty:
        return "[Empty table]"

    text_lines: list[str] = []

    # Column headers
    headers = [str(col) for col in df.columns]
    header_line = " | ".join(headers)
    text_lines.append(header_line)
    text_lines.append("-" * len(header_line))

    # Data rows
    for _, row in df.iterrows():
        row_values = [str(val) if pd.notna(val) else "" for val in row]
        row_line = " | ".join(row_values)
        text_lines.append(row_line)

    return "\n".join(text_lines)


@tool
async def read_table(
    image_path: str,
    language: str = "eng",
    borderless_tables: bool = True,
    implicit_rows: bool = True,
    export_xlsx: str | None = None,
) -> str:
    """Extract table data from image files and convert to structured format (CSV/Excel), supports both bordered and borderless tables.

    Args:
        image_path: Path to the image file containing tables (supports relative and absolute paths).
        language: OCR language code: 'eng' (English), 'chi_sim' (Simplified Chinese), 'chi_tra' (Traditional Chinese), etc.
        borderless_tables: Whether to detect borderless tables.
        implicit_rows: Whether to infer implicit rows through alignment.
        export_xlsx: Optional path to export all tables to Excel file (e.g., 'tables.xlsx').
    """
    try:
        image_file = Path(image_path)
        if not image_file.exists():
            return f"Image file does not exist: {image_path}"

        if image_file.suffix.lower() not in _ALLOWED_EXTENSIONS:
            return (
                f"Unsupported image format: {image_file.suffix}. "
                f"Supported formats: {', '.join(_ALLOWED_EXTENSIONS)}"
            )

        # Import img2table lazily — the heavy backends are optional at
        # bind-time (bind_tools schema checks must not require them).
        try:
            from img2table.document import Image as _Img2TableImage
            from img2table.ocr import TesseractOCR as _TesseractOCR
        except ImportError as e:
            return (
                "Missing required dependencies. Please install: "
                f"pip install img2table tesseract. Error: {str(e)}"
            )

        doc = _Img2TableImage(str(image_file))

        try:
            ocr_engine = _TesseractOCR(lang=language)
        except Exception as e:  # noqa: BLE001
            return (
                f"Failed to initialize OCR engine with language '{language}': {str(e)}"
            )

        tables = doc.extract_tables(
            ocr=ocr_engine,
            borderless_tables=borderless_tables,
            implicit_rows=implicit_rows,
        )

        if not tables:
            return "No tables detected in the image."

        table_results: list[dict] = []
        dfs: list[pd.DataFrame] = []

        for i, table in enumerate(tables, start=1):
            df = table.df
            dfs.append(df)
            table_results.append(
                {
                    "table_number": i,
                    "rows": df.shape[0],
                    "columns": df.shape[1],
                    "data": _format_table_as_text(df),
                }
            )

        excel_export_info = ""
        if export_xlsx and dfs:
            try:
                with pd.ExcelWriter(export_xlsx) as writer:
                    for i, df in enumerate(dfs, start=1):
                        sheet_name = f"Table_{i}"
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                excel_export_info = f"All tables exported to Excel: {export_xlsx}"
            except Exception as e:  # noqa: BLE001
                excel_export_info = f"Excel export failed: {str(e)}"

        output_text = (
            "Table Extraction Results:\n"
            f"Found {len(tables)} table(s) in the image.\n\n"
        )

        for table_info in table_results:
            output_text += (
                f"Table #{table_info['table_number']} "
                f"({table_info['rows']} rows × {table_info['columns']} columns):\n"
                f"{table_info['data']}\n\n"
            )

        if excel_export_info:
            output_text += f"{excel_export_info}\n"

        return output_text.strip()

    except Exception as e:  # noqa: BLE001
        return f"Table extraction error: {str(e)}"
