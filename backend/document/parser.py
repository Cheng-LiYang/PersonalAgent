"""Parsers for supported local document formats."""
from pathlib import Path
from backend.models import DocumentPage

def category_for(path: Path, root: Path) -> str:
    """Derive category from the first directory below the knowledge root."""
    relative = path.resolve().relative_to(root.resolve())
    return relative.parts[0] if len(relative.parts) > 1 else "未分类"

def parse_pdf(path: Path, root: Path) -> list[DocumentPage]:
    """Extract PDF pages and their page numbers with PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF support requires PyMuPDF") from exc
    result = []
    with fitz.open(path) as document:
        title = document.metadata.get("title") or path.stem
        for index, page in enumerate(document):
            text = page.get_text("text").strip()
            if text:
                result.append(DocumentPage(text, index + 1, str(path.resolve()), title, category_for(path, root)))
    return result

def parse_docx(path: Path, root: Path) -> list[DocumentPage]:
    """Extract DOCX paragraphs as one logical page."""
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("DOCX support requires python-docx") from exc
    text = "\n".join(p.text for p in Document(path).paragraphs if p.text.strip())
    return [DocumentPage(text, 1, str(path.resolve()), path.stem, category_for(path, root))] if text else []

def parse_text(path: Path, root: Path) -> list[DocumentPage]:
    """Read UTF-8 Markdown or plain text."""
    text = path.read_text(encoding="utf-8-sig").strip()
    return [DocumentPage(text, 1, str(path.resolve()), path.stem, category_for(path, root))] if text else []

def parse_document(path: Path, root: Path) -> list[DocumentPage]:
    """Dispatch to a parser based on extension."""
    if path.suffix.lower() == ".pdf":
        return parse_pdf(path, root)
    if path.suffix.lower() == ".docx":
        return parse_docx(path, root)
    if path.suffix.lower() in {".md", ".txt"}:
        return parse_text(path, root)
    raise ValueError(f"Unsupported document: {path.suffix}")
