from typing import List
from pathlib import Path
from loguru import logger

from .base_loader import BaseLoader, Document
from .pdf_loader import PDFLoader
from .text_loader import TextLoader
from .docx_loader import DocxLoader
from .web_loader import WebLoader
from .json_loader import JSONLoader


class LoaderFactory:
    """
    Auto-detect loader yang tepat berdasarkan ekstensi file atau URL.
    Pattern: Factory Method — client tidak perlu tahu loader mana yang dipakai.
    """

    _loaders: dict[str, BaseLoader] = {
        ".pdf": PDFLoader(),
        ".txt": TextLoader(),
        ".md": TextLoader(),
        ".markdown": TextLoader(),
        ".docx": DocxLoader(),
        ".doc": DocxLoader(),
        ".json": JSONLoader(),
        ".jsonl": JSONLoader(),
    }

    @classmethod
    def get_loader(cls, source: str) -> BaseLoader:
        """Pilih loader yang sesuai untuk source."""
        # URL
        if source.startswith(("http://", "https://")):
            return WebLoader()

        # File
        ext = Path(source).suffix.lower()
        loader = cls._loaders.get(ext)
        if loader is None:
            raise ValueError(
                f"Tidak ada loader untuk ekstensi '{ext}'. "
                f"Didukung: {list(cls._loaders.keys())} + URL"
            )
        return loader

    @classmethod
    def load(cls, source: str) -> List[Document]:
        """One-liner: auto-detect loader dan langsung load."""
        loader = cls.get_loader(source)
        logger.info(f"Using {loader.__class__.__name__} for: {source}")
        return loader.load(source)

    @classmethod
    def load_many(cls, sources: List[str]) -> List[Document]:
        """Load multiple sources sekaligus."""
        all_docs = []
        for source in sources:
            try:
                docs = cls.load(source)
                all_docs.extend(docs)
                logger.info(f"Loaded {len(docs)} docs from {source}")
            except Exception as e:
                logger.error(f"Gagal load {source}: {e}")
        logger.info(f"Total loaded: {len(all_docs)} documents")
        return all_docs


__all__ = ["LoaderFactory", "Document", "BaseLoader"]
