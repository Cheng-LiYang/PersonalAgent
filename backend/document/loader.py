"""Safe recursive knowledge-base discovery."""
import logging
from pathlib import Path
from backend.document.parser import parse_document
from backend.models import DocumentPage

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx"}
DEFAULT_EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", "build", "dist", "dist_updated",
}

class DocumentLoader:
    """Load supported files from one explicitly selected root."""
    def __init__(self, root: str | Path, excluded_directories: list[str] | set[str] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        configured = excluded_directories or DEFAULT_EXCLUDED_DIRECTORIES
        self.excluded_directories = {name.casefold() for name in configured}

    def scan(self) -> list[Path]:
        if not self.root.is_dir():
            raise NotADirectoryError(str(self.root))
        files = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative = path.relative_to(self.root)
            if any(part.casefold() in self.excluded_directories for part in relative.parts[:-1]):
                continue
            files.append(path)
        return sorted(files)

    def load(self) -> tuple[list[DocumentPage], list[str]]:
        pages: list[DocumentPage] = []
        errors: list[str] = []
        for path in self.scan():
            try:
                pages.extend(parse_document(path, self.root))
            except (OSError, RuntimeError, ValueError) as exc:
                LOGGER.exception("Unable to parse %s", path)
                errors.append(f"{path}: {exc}")
        return pages, errors
