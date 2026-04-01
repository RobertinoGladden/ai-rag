from __future__ import annotations

import io
import base64
from pathlib import Path
from typing import Union
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ExifTags
from loguru import logger


@dataclass
class ImageInput:
    """Normalized image container — semua sumber dikonversi ke sini."""
    pil_image: Image.Image
    original_size: tuple[int, int]   # (width, height)
    source: str = "unknown"
    filename: str = ""
    format: str = "RGB"
    metadata: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.pil_image.width

    @property
    def height(self) -> int:
        return self.pil_image.height

    @property
    def numpy(self) -> np.ndarray:
        """Return as HWC uint8 numpy array (untuk OpenCV/YOLO)."""
        return np.array(self.pil_image)

    def to_base64(self) -> str:
        """Konversi ke base64 string untuk response API."""
        buf = io.BytesIO()
        self.pil_image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()


class ImagePreprocessor:
    """
    Handle semua bentuk input gambar:
    - File path (str / Path)
    - Raw bytes (dari upload)
    - Base64 string
    - URL (via HTTP)
    - PIL Image langsung
    """

    MAX_SIZE = (1920, 1920)

    @classmethod
    def load(cls, source: Union[str, bytes, Path, Image.Image]) -> ImageInput:
        """Auto-detect tipe input dan load sebagai ImageInput."""
        if isinstance(source, Image.Image):
            return cls._from_pil(source, source_name="pil_direct")

        if isinstance(source, bytes):
            return cls._from_bytes(source)

        if isinstance(source, Path) or (isinstance(source, str) and not source.startswith(("http", "data:"))):
            return cls._from_file(str(source))

        if isinstance(source, str) and source.startswith("data:image"):
            return cls._from_base64(source)

        if isinstance(source, str) and source.startswith(("http://", "https://")):
            return cls._from_url(source)

        raise ValueError(f"Tipe input tidak dikenali: {type(source)}")

    @classmethod
    def _from_file(cls, path: str) -> ImageInput:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Gambar tidak ditemukan: {path}")
        img = Image.open(p)
        img = cls._normalize(img)
        logger.debug(f"Loaded image from file: {p.name} ({img.width}x{img.height})")
        return ImageInput(
            pil_image=img,
            original_size=(img.width, img.height),
            source="file",
            filename=p.name,
            metadata={"path": str(p), "format": p.suffix},
        )

    @classmethod
    def _from_bytes(cls, data: bytes, filename: str = "upload") -> ImageInput:
        img = Image.open(io.BytesIO(data))
        original_size = (img.width, img.height)
        img = cls._normalize(img)
        return ImageInput(
            pil_image=img,
            original_size=original_size,
            source="bytes",
            filename=filename,
            metadata={"size_bytes": len(data)},
        )

    @classmethod
    def _from_base64(cls, b64_str: str) -> ImageInput:
        # Strip data URI prefix jika ada
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        data = base64.b64decode(b64_str)
        return cls._from_bytes(data, filename="base64_input")

    @classmethod
    def _from_url(cls, url: str) -> ImageInput:
        import httpx
        logger.debug(f"Fetching image from URL: {url}")
        r = httpx.get(url, timeout=15, follow_redirects=True)
        r.raise_for_status()
        img_input = cls._from_bytes(r.content, filename=url.split("/")[-1] or "url_image")
        img_input.source = "url"
        img_input.metadata["url"] = url
        return img_input

    @classmethod
    def _from_pil(cls, img: Image.Image, source_name: str = "pil") -> ImageInput:
        original_size = (img.width, img.height)
        img = cls._normalize(img)
        return ImageInput(pil_image=img, original_size=original_size, source=source_name)

    @classmethod
    def _normalize(cls, img: Image.Image) -> Image.Image:
        """Convert ke RGB, fix EXIF rotation, resize jika terlalu besar."""
        # Fix EXIF orientation
        try:
            exif = img._getexif()
            if exif:
                for tag, val in exif.items():
                    if ExifTags.TAGS.get(tag) == "Orientation":
                        rotations = {3: 180, 6: 270, 8: 90}
                        if val in rotations:
                            img = img.rotate(rotations[val], expand=True)
        except Exception:
            pass

        # Convert ke RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize jika melebihi batas
        if img.width > cls.MAX_SIZE[0] or img.height > cls.MAX_SIZE[1]:
            img.thumbnail(cls.MAX_SIZE, Image.LANCZOS)
            logger.debug(f"Resized image to {img.width}x{img.height}")

        return img
