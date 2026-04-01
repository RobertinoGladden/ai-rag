from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class CVSettings(BaseSettings):
    # Device
    device: str = Field("cpu", env="CV_DEVICE")           # "cpu" atau "cuda"

    # CLIP
    clip_model: str = Field("ViT-B-32", env="CLIP_MODEL")
    clip_pretrained: str = Field("openai", env="CLIP_PRETRAINED")

    # YOLO
    yolo_model: str = Field("yolov8n.pt", env="YOLO_MODEL")  # n=nano, s=small, m=medium
    yolo_conf_threshold: float = Field(0.25, env="YOLO_CONF")
    yolo_iou_threshold: float = Field(0.45, env="YOLO_IOU")

    # Image Captioning
    caption_model: str = Field(
        "Salesforce/blip-image-captioning-base", env="CAPTION_MODEL"
    )

    # OCR
    ocr_engine: str = Field("easyocr", env="OCR_ENGINE")    # "easyocr" atau "tesseract"
    ocr_languages: str = Field("en,id", env="OCR_LANGUAGES") # comma-separated

    # API
    api_host: str = Field("0.0.0.0", env="CV_API_HOST")
    api_port: int = Field(8001, env="CV_API_PORT")
    max_image_size_mb: float = Field(10.0, env="MAX_IMAGE_SIZE_MB")

    # Storage
    upload_dir: str = Field("./uploads", env="CV_UPLOAD_DIR")
    models_cache_dir: str = Field("./model_cache", env="CV_MODELS_CACHE")

    # MLflow
    mlflow_tracking_uri: str = Field("./mlruns", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field("cv_pipeline", env="MLFLOW_CV_EXPERIMENT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        for d in [self.upload_dir, self.models_cache_dir, "./logs", "./mlruns"]:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_cv_settings() -> CVSettings:
    s = CVSettings()
    s.ensure_dirs()
    return s
