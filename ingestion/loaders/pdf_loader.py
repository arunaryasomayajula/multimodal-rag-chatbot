from pathlib import Path
import fitz  # pymupdf
import pdfplumber
from ingestion.loaders.text_loader import RawChunk


def load_pdf(path: str | Path) -> list[RawChunk]:
    path = Path(path)
    chunks: list[RawChunk] = []

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                text = _ocr_page(page)
            if text:
                chunks.append(RawChunk(
                    content=text,
                    metadata={
                        "source_file": path.name,
                        "page_number": page_num,
                        "modality": "pdf",
                    },
                ))

    # extract embedded tables via pdfplumber
    with pdfplumber.open(str(path)) as doc:
        for page_num, page in enumerate(doc.pages, start=1):
            for table in page.extract_tables():
                rows = [
                    " | ".join(str(c or "").strip() for c in row)
                    for row in table if any(row)
                ]
                if rows:
                    chunks.append(RawChunk(
                        content="\n".join(rows),
                        metadata={
                            "source_file": path.name,
                            "page_number": page_num,
                            "modality": "pdf_table",
                        },
                    ))

    return chunks


def _ocr_page(page: fitz.Page) -> str:
    try:
        import pytesseract
        from PIL import Image
        import io

        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception:
        return ""
