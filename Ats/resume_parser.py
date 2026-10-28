from io import BytesIO
import logging
import re
from docx import Document
from docx.oxml.ns import qn
from pypdf import PdfReader

logger = logging.getLogger(__name__)
MAX_FILE_SIZE = 20 * 1024 * 1024

def clean_extracted_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()

def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))

    if reader.is_encrypted:
        result = reader.decrypt("")
        if result == 0:
            raise ValueError("This PDF is password protected.")

    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        standard_text = ""
        layout_text = ""

        try:
            standard_text = page.extract_text() or ""
        except Exception:
            logger.exception("Standard extraction failed on page %s", page_number)

        try:
            layout_text = page.extract_text(extraction_mode="layout") or ""
        except Exception:
            logger.exception("Layout extraction failed on page %s", page_number)

        standard_clean = clean_extracted_text(standard_text)
        layout_clean = clean_extracted_text(layout_text)

        page_text = (
            layout_clean
            if len(layout_clean) > len(standard_clean) * 1.2
            else standard_clean
        )

        if page_text:
            extracted_pages.append(page_text)

    return "\n\n".join(extracted_pages)

def extract_docx_text(file_bytes: bytes) -> str:
    document = Document(BytesIO(file_bytes))
    parts = []

    for element in document.element.body.iterchildren():
        if element.tag == qn("w:p"):
            text_parts = [
                node.text
                for node in element.iter(qn("w:t"))
                if node.text
            ]
            text = "".join(text_parts).strip()
            if text:
                parts.append(text)

        elif element.tag == qn("w:tbl"):
            for row in element.iter(qn("w:tr")):
                cells = []

                for cell in row.iter(qn("w:tc")):
                    cell_text = [
                        node.text
                        for node in cell.iter(qn("w:t"))
                        if node.text
                    ]

                    value = " ".join(
                        item.strip()
                        for item in cell_text
                        if item.strip()
                    )

                    if value:
                        cells.append(value)

                if cells:
                    parts.append(" | ".join(cells))

    return "\n".join(parts)

def extract_resume_text(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError("File exceeds the 20 MB limit.")

    if filename.endswith(".pdf"):
        text = extract_pdf_text(file_bytes)
    elif filename.endswith(".docx"):
        text = extract_docx_text(file_bytes)
    else:
        raise ValueError("Unsupported file format.")

    text = clean_extracted_text(text)

    if not text:
        raise ValueError(
            "No readable text could be extracted. "
            "The PDF may contain scanned images. OCR may be required."
        )

    return text