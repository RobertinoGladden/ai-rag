from typing import List
from pathlib import Path
from loguru import logger

from .base_loader import BaseLoader, Document


class PDFLoader(BaseLoader):
    """Loader untuk file PDF menggunakan pypdf."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    def load(self, source: str) -> List[Document]:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {source}")

        logger.info(f"Loading PDF: {path.name}")
        reader = PdfReader(str(path))
        documents = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            documents.append(Document(
                content=text.strip(),
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "page": i + 1,
                    "total_pages": len(reader.pages),
                    "type": "pdf",
                }
            ))

        logger.info(f"Loaded {len(documents)} pages from {path.name}")
        return documents
