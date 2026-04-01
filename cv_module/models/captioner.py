from __future__ import annotations

from dataclasses import dataclass
from loguru import logger

from ..config import get_cv_settings
from ..processors.image_preprocessor import ImageInput


@dataclass
class CaptionResult:
    caption: str
    model: str
    confidence: float = 1.0


class ImageCaptioner:
    """
    Image captioning menggunakan BLIP (Bootstrapped Language-Image Pre-training).
    Model Salesforce/blip-image-captioning-base — ringan, akurat, bisa jalan di CPU.

    Output: deskripsi teks natural dari gambar.
    Berguna untuk: accessibility, content indexing, multimodal RAG.
    """

    def __init__(self):
        settings = get_cv_settings()
        logger.info(f"Loading captioning model: {settings.caption_model}")

        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch

            self.device = settings.device
            self.processor = BlipProcessor.from_pretrained(
                settings.caption_model,
                cache_dir=settings.models_cache_dir,
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                settings.caption_model,
                cache_dir=settings.models_cache_dir,
            ).to(self.device)
            self.model.eval()
            self.model_name = settings.caption_model
            logger.info("Image captioner ready.")

        except Exception as e:
            logger.error(f"Gagal load captioning model: {e}")
            raise

    def caption(
        self,
        image: ImageInput,
        prompt: str = None,
        max_new_tokens: int = 100,
    ) -> CaptionResult:
        """
        Generate caption untuk gambar.

        Args:
            image: ImageInput object
            prompt: Optional — beri konteks/instruksi, e.g. "a photo of"
            max_new_tokens: Panjang maksimum caption

        Returns:
            CaptionResult berisi caption string
        """
        import torch

        if prompt:
            inputs = self.processor(
                image.pil_image, prompt, return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                image.pil_image, return_tensors="pt"
            ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )

        caption = self.processor.decode(output[0], skip_special_tokens=True)

        # Bersihkan prefix prompt dari output
        if prompt and caption.lower().startswith(prompt.lower()):
            caption = caption[len(prompt):].strip()

        logger.debug(f"Caption: {caption}")

        return CaptionResult(
            caption=caption,
            model=self.model_name,
        )

    def visual_qa(self, image: ImageInput, question: str) -> CaptionResult:
        """
        Visual Question Answering — tanya tentang isi gambar.
        Contoh: "What color is the car?" → "red"
        """
        return self.caption(image, prompt=question, max_new_tokens=50)
