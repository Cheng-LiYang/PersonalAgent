"""End-to-end knowledge-base indexing service."""
from pathlib import Path
import threading
from typing import Any
from collections.abc import Callable
from backend.document import DocumentLoader, RecursiveTextSplitter
from backend.document.parser import parse_document
from backend.embedding import create_embedding_model
from backend.retrieval.vector_store import VectorStore

class IndexService:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        data_dir = Path(config["app"]["data_dir"])
        self.store = VectorStore(data_dir / "vectors.sqlite3", create_embedding_model(config["embedding"]))
        self._build_lock = threading.Lock()

    def build(self, knowledge_base: str | Path | None = None, progress: Callable[[str, int, str], None] | None = None) -> dict[str, Any]:
        """Serialize index mutation within this process while reads continue via WAL."""

        with self._build_lock:
            return self._build(knowledge_base, progress)

    def _build(self, knowledge_base: str | Path | None = None, progress: Callable[[str, int, str], None] | None = None) -> dict[str, Any]:
        root = Path(knowledge_base or self.config["app"]["knowledge_base"])
        report = progress or (lambda phase, percent, detail: None)
        report("scan", 2, "正在扫描知识库目录…")
        loader = DocumentLoader(root, self.config["document"].get("excluded_directories"))
        files = loader.scan()
        pages, errors, failed_paths = [], [], set()
        for index, path in enumerate(files, 1):
            report("parse", 5 + int(35 * index / max(1, len(files))), f"正在解析 {path.name}（{index}/{len(files)}）")
            try:
                pages.extend(parse_document(path, root))
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"{path}: {exc}")
                failed_paths.add(str(path.resolve()))
        settings = self.config["document"]
        splitter = RecursiveTextSplitter(settings["chunk_size"], settings["chunk_overlap"])
        chunks = []
        for index, page in enumerate(pages, 1):
            report("split", 40 + int(20 * index / max(1, len(pages))), f"正在切分 {Path(page.source_path).name} · 第 {page.page} 页")
            chunks.extend(splitter.split_page(page))
        report("embedding", 61, f"正在检查 {len(chunks)} 个切片的增量变化…")
        changes = self.store.sync(
            chunks,
            lambda current, total: report("embedding", 61 + int(28 * current / max(1, total)), f"正在生成新增或变更切片的 Embedding（{current}/{total}）"),
            lambda: report("storage", 92, "正在同步增量索引…"),
            failed_paths,
        )
        report("storage", 99, "正在刷新关键词索引…")
        return {"files": len({page.source_path for page in pages}), "pages": len(pages), "chunks": len(chunks), **changes, "errors": errors}
