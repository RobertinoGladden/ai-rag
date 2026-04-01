from typing import List
from pathlib import Path
from loguru import logger

from .base_loader import BaseLoader, Document


class TextLoader(BaseLoader):
    """Loader untuk file .txt dan .md."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".txt", ".md", ".markdown"]

    def load(self, source: str) -> List[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {source}")

        logger.info(f"Loading text file: {path.name}")
        content = path.read_text(encoding="utf-8")

        return [Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "type": path.suffix.lstrip("."),
                "size_chars": len(content),
            }
        )]
