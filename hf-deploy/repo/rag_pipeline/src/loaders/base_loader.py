from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Document:
    """Representasi satu dokumen atau chunk yang sudah diload."""
    content: str
    metadata: dict = field(default_factory=dict)
    doc_id: Optional[str] = None

    def __post_init__(self):
        if self.doc_id is None:
            import hashlib
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class BaseLoader(ABC):
    """Abstract base class untuk semua document loaders."""

    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        Load dokumen dari source (path file atau URL).
        Returns list of Document objects.
        """
        pass

    def validate_source(self, source: str) -> bool:
        """Validasi apakah source bisa di-handle loader ini."""
        return True

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Daftar ekstensi file yang didukung loader ini."""
        pass
