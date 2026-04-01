import json
from typing import List
from pathlib import Path
from loguru import logger

from .base_loader import BaseLoader, Document


class JSONLoader(BaseLoader):
    """
    Loader untuk file JSON.
    Bisa flatten nested JSON menjadi teks untuk di-embed.
    """

    def __init__(self, text_key: str = None, jq_schema: str = None):
        """
        text_key: key spesifik yang jadi konten utama (e.g. 'content', 'text')
        jq_schema: opsional — filter JSON pakai jq-style path
        """
        self.text_key = text_key
        self.jq_schema = jq_schema

    @property
    def supported_extensions(self) -> List[str]:
        return [".json", ".jsonl"]

    def load(self, source: str) -> List[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {source}")

        logger.info(f"Loading JSON: {path.name}")

        # Handle JSONL (JSON Lines)
        if path.suffix == ".jsonl":
            return self._load_jsonl(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Jika list of records
        if isinstance(data, list):
            documents = []
            for i, record in enumerate(data):
                content = self._extract_content(record)
                documents.append(Document(
                    content=content,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "type": "json",
                        "record_index": i,
                    }
                ))
            return documents

        # Single object
        content = self._extract_content(data)
        return [Document(
            content=content,
            metadata={
                "source": str(path),
                "filename": path.name,
                "type": "json",
            }
        )]

    def _load_jsonl(self, path: Path) -> List[Document]:
        documents = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                content = self._extract_content(record)
                documents.append(Document(
                    content=content,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "type": "jsonl",
                        "line": i + 1,
                    }
                ))
        return documents

    def _extract_content(self, data: dict) -> str:
        """Konversi dict/list ke string yang bisa di-embed."""
        if self.text_key and isinstance(data, dict) and self.text_key in data:
            return str(data[self.text_key])

        # Fallback: flatten semua key-value pair
        if isinstance(data, dict):
            parts = []
            for k, v in data.items():
                if isinstance(v, (str, int, float, bool)):
                    parts.append(f"{k}: {v}")
                elif isinstance(v, (list, dict)):
                    parts.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
            return "\n".join(parts)

        return json.dumps(data, ensure_ascii=False, indent=2)
