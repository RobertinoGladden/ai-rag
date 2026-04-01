from __future__ import annotations

from typing import List
from dataclasses import dataclass
from functools import lru_cache

import torch
import open_clip
from loguru import logger

from ..config import get_cv_settings
from ..processors.image_preprocessor import ImageInput


@dataclass
class CLIPResult:
    """Hasil dari CLIP model."""
    # Zero-shot classification
    labels: List[str] = None
    probabilities: List[float] = None
    top_label: str = ""
    top_score: float = 0.0

    # Image-text similarity
    similarity_score: float = None

    # Image features (untuk downstream tasks)
    image_features: "torch.Tensor" = None


class CLIPModel:
    """
    Wrapper CLIP menggunakan open_clip.
    Capabilities:
    - Zero-shot image classification (tanpa training!)
    - Image-text similarity scoring
    - Image feature extraction untuk retrieval
    """

    def __init__(self):
        settings = get_cv_settings()
        self.device = settings.device
        logger.info(f"Loading CLIP model: {settings.clip_model} ({settings.clip_pretrained})")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model,
            pretrained=settings.clip_pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model)
        self.model.eval()
        logger.info("CLIP model ready.")

    @torch.no_grad()
    def classify(self, image: ImageInput, labels: List[str]) -> CLIPResult:
        """
        Zero-shot classification — tentukan kategori gambar dari daftar label.
        Tidak perlu training sama sekali!

        Args:
            image: ImageInput object
            labels: List label kandidat, e.g. ["kucing", "anjing", "burung"]

        Returns:
            CLIPResult dengan probabilitas tiap label
        """
        # Preprocess image
        img_tensor = self.preprocess(image.pil_image).unsqueeze(0).to(self.device)

        # Tokenize labels
        text_tokens = self.tokenizer(labels).to(self.device)

        # Compute features
        image_features = self.model.encode_image(img_tensor)
        text_features = self.model.encode_text(text_tokens)

        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity (cosine similarity → softmax → probabilities)
        logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = logits[0].cpu().numpy().tolist()

        top_idx = int(torch.argmax(logits[0]).item())

        return CLIPResult(
            labels=labels,
            probabilities=[round(p, 4) for p in probs],
            top_label=labels[top_idx],
            top_score=round(probs[top_idx], 4),
        )

    @torch.no_grad()
    def compute_similarity(self, image: ImageInput, text: str) -> float:
        """
        Hitung seberapa relevan teks dengan gambar (0.0 - 1.0).
        Berguna untuk: image search, content moderation, caption scoring.
        """
        img_tensor = self.preprocess(image.pil_image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer([text]).to(self.device)

        img_feat = self.model.encode_image(img_tensor)
        txt_feat = self.model.encode_text(text_tokens)

        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

        similarity = (img_feat @ txt_feat.T).item()

        # Normalize ke 0-1 (CLIP output biasanya -1 to 1)
        return round((similarity + 1) / 2, 4)

    @torch.no_grad()
    def extract_features(self, image: ImageInput) -> "torch.Tensor":
        """
        Ekstrak image embedding untuk semantic image search / clustering.
        Output: tensor shape (512,) untuk ViT-B-32
        """
        img_tensor = self.preprocess(image.pil_image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(img_tensor)
        features /= features.norm(dim=-1, keepdim=True)
        return features[0].cpu()

    @torch.no_grad()
    def rank_images_by_text(
        self,
        images: List[ImageInput],
        query_text: str,
    ) -> List[tuple[int, float]]:
        """
        Rank multiple images berdasarkan relevansi dengan teks query.
        Returns: list of (original_index, score) sorted by score desc.
        Berguna untuk: text-to-image search.
        """
        tensors = torch.stack([
            self.preprocess(img.pil_image) for img in images
        ]).to(self.device)

        text_tokens = self.tokenizer([query_text]).to(self.device)

        img_features = self.model.encode_image(tensors)
        txt_features = self.model.encode_text(text_tokens)

        img_features /= img_features.norm(dim=-1, keepdim=True)
        txt_features /= txt_features.norm(dim=-1, keepdim=True)

        scores = (img_features @ txt_features.T).squeeze(1).cpu().numpy()
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
        return [(idx, round(score, 4)) for idx, score in ranked]
