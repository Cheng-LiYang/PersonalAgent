"""Metadata-preserving recursive character splitter."""
import hashlib
from pathlib import Path
from backend.models import Chunk, DocumentPage

class RecursiveTextSplitter:
    """Split text at natural boundaries with a bounded overlap."""
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Require chunk_size > chunk_overlap >= 0")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_page(self, page: DocumentPage) -> list[Chunk]:
        text, chunks, start = page.text.strip(), [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                boundaries = [text.rfind(sep, start + self.chunk_size // 2, end) for sep in ("\n\n", "\n", "。", ". ", " ")]
                natural = max(boundaries)
                if natural > start:
                    end = natural + 1
            content = text[start:end].strip()
            digest = hashlib.sha256(f"{page.source_path}:{page.page}:{start}:{content}".encode()).hexdigest()[:24]
            if content:
                chunks.append(Chunk(digest, content, Path(page.source_path).name, page.source_path, page.page, page.category, page.title))
            if end >= len(text):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return chunks

    def split(self, pages: list[DocumentPage]) -> list[Chunk]:
        return [chunk for page in pages for chunk in self.split_page(page)]
