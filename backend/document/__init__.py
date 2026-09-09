"""Document discovery, parsing, and splitting."""
from .loader import DocumentLoader
from .splitter import RecursiveTextSplitter
__all__ = ["DocumentLoader", "RecursiveTextSplitter"]
