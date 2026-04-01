import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from PIL import Image
import io


# === Fixtures ===

@pytest.fixture
def sample_image() -> Image.Image:
    """Buat gambar dummy 224x224 RGB untuk testing."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sample_image_bytes(sample_image) -> bytes:
    buf = io.BytesIO()
    sample_image.save(buf, format="JPEG")
    return buf.getvalue()


# === ImagePreprocessor tests ===

def test_preprocessor_from_pil(sample_image):
    from src.processors.image_preprocessor import ImagePreprocessor
    result = ImagePreprocessor.load(sample_image)
    assert result.pil_image is not None
    assert result.width == 224
    assert result.height == 224
    assert result.source == "pil_direct"


def test_preprocessor_from_bytes(sample_image_bytes):
    from src.processors.image_preprocessor import ImagePreprocessor
    result = ImagePreprocessor.load(sample_image_bytes)
    assert result.source == "bytes"
    assert result.pil_image.mode == "RGB"


def test_preprocessor_converts_to_rgb():
    from src.processors.image_preprocessor import ImagePreprocessor
    rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
    result = ImagePreprocessor.load(rgba_img)
    assert result.pil_image.mode == "RGB"


def test_preprocessor_resize_large_image():
    from src.processors.image_preprocessor import ImagePreprocessor
    large_img = Image.new("RGB", (3000, 3000), (100, 100, 100))
    result = ImagePreprocessor.load(large_img)
    assert result.width <= 1920
    assert result.height <= 1920


def test_preprocessor_numpy_output(sample_image):
    from src.processors.image_preprocessor import ImagePreprocessor
    result = ImagePreprocessor.load(sample_image)
    arr = result.numpy
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.uint8


def test_preprocessor_file_not_found():
    from src.processors.image_preprocessor import ImagePreprocessor
    with pytest.raises(FileNotFoundError):
        ImagePreprocessor.load("/tidak/ada/gambar.jpg")


def test_preprocessor_to_base64(sample_image):
    from src.processors.image_preprocessor import ImagePreprocessor
    result = ImagePreprocessor.load(sample_image)
    b64 = result.to_base64()
    assert isinstance(b64, str)
    assert len(b64) > 0


# === CLIP model tests (mocked) ===

@patch("src.models.clip_model.open_clip")
def test_clip_classify(mock_clip):
    import torch
    from src.models.clip_model import CLIPModel
    from src.processors.image_preprocessor import ImageInput

    mock_model = MagicMock()
    mock_clip.create_model_and_transforms.return_value = (
        mock_model, MagicMock(), MagicMock()
    )
    mock_clip.get_tokenizer.return_value = MagicMock()

    # Mock feature outputs
    feat = torch.randn(1, 512)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    mock_model.encode_image.return_value = feat.clone()
    mock_model.encode_text.return_value = feat.clone()

    clip = CLIPModel()
    img_input = MagicMock(spec=["pil_image"])
    img_input.pil_image = Image.new("RGB", (224, 224))

    result = clip.classify(img_input, ["kucing", "anjing", "burung"])

    assert result.labels == ["kucing", "anjing", "burung"]
    assert len(result.probabilities) == 3
    assert abs(sum(result.probabilities) - 1.0) < 0.01
    assert result.top_label in ["kucing", "anjing", "burung"]


# === YOLO detector tests (mocked) ===

@patch("src.models.yolo_detector.YOLO")
def test_yolo_detect_empty(MockYOLO):
    from src.models.yolo_detector import YOLODetector
    from src.processors.image_preprocessor import ImageInput
    import numpy as np

    mock_yolo = MockYOLO.return_value
    mock_result = MagicMock()
    mock_result.boxes = None
    mock_yolo.predict.return_value = [mock_result]
    mock_yolo.names = {0: "person", 1: "car"}

    detector = YOLODetector()
    img = MagicMock(spec=["numpy", "width", "height"])
    img.numpy = np.zeros((224, 224, 3), dtype=np.uint8)
    img.width = 224
    img.height = 224

    result = detector.detect(img)
    assert result.count == 0
    assert result.labels_summary == {}


def test_detection_result_summary():
    from src.models.yolo_detector import DetectionResult, Detection, BoundingBox

    dets = [
        Detection("person", 0.9, BoundingBox(0, 0, 100, 200), 0),
        Detection("person", 0.8, BoundingBox(10, 10, 80, 180), 0),
        Detection("car", 0.75, BoundingBox(200, 100, 400, 300), 1),
    ]
    result = DetectionResult(detections=dets, image_width=640, image_height=480)

    assert result.count == 3
    assert result.labels_summary == {"person": 2, "car": 1}
    assert len(result.filter_by_label("person")) == 2
    assert len(result.filter_by_confidence(0.85)) == 1


# === CVAnalysisResult tests ===

def test_cv_result_to_summary():
    from src.cv_pipeline import CVAnalysisResult
    from src.models.captioner import CaptionResult
    from src.models.yolo_detector import DetectionResult, Detection, BoundingBox

    result = CVAnalysisResult(
        image_width=640,
        image_height=480,
        caption=CaptionResult(caption="a cat sitting on a table", model="blip"),
        detections=DetectionResult(detections=[
            Detection("cat", 0.92, BoundingBox(100, 100, 300, 300), 15)
        ], image_width=640, image_height=480),
    )

    summary = result.to_summary()
    assert "cat sitting on a table" in summary
    assert "1x cat" in summary


def test_cv_result_empty_summary():
    from src.cv_pipeline import CVAnalysisResult
    result = CVAnalysisResult()
    assert "Tidak ada informasi" in result.to_summary()


# === API integration tests (mocked) ===

@patch("src.api.routes.CVPipeline")
def test_api_health(MockPipeline):
    from fastapi.testclient import TestClient
    from src.api.main import app

    client = TestClient(app)
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
