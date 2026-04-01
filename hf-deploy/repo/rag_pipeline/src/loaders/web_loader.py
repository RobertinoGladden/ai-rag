from typing import List
from loguru import logger

from .base_loader import BaseLoader, Document


class WebLoader(BaseLoader):
    """Loader untuk URL — scrape konten teks dari halaman web."""

    @property
    def supported_extensions(self) -> List[str]:
        return []  # Tidak berbasis ekstensi, berbasis URL

    def validate_source(self, source: str) -> bool:
        return source.startswith(("http://", "https://"))

    def load(self, source: str) -> List[Document]:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Install: pip install requests beautifulsoup4")

        logger.info(f"Fetching URL: {source}")
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Pipeline/1.0)"}

        response = requests.get(source, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Hapus tag yang tidak relevan
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Ambil judul
        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else ""

        # Ambil konten utama
        main = soup.find("main") or soup.find("article") or soup.find("body")
        content = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

        # Bersihkan baris kosong berulang
        lines = [line for line in content.splitlines() if line.strip()]
        content = "\n".join(lines)

        return [Document(
            content=content,
            metadata={
                "source": source,
                "title": title_text,
                "type": "web",
                "status_code": response.status_code,
                "content_length": len(content),
            }
        )]
