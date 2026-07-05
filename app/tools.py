from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader


class QwenVisionTool:
    """A deterministic, local-only document understanding tool.

    This implementation uses local parsing for PDF and image files and does not
    attempt remote Qwen or Ollama calls.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "deterministic-local")

    def invoke(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                text = self._extract_pdf_text(path)
            except Exception as exc:  # pragma: no cover - defensive fallback for demo purposes
                text = f"Unable to read PDF content: {exc}"
            return {
                "summary": "Extracted text from PDF using deterministic local flow",
                "extracted_text": text,
                "source": str(path),
            }

        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            text = self._extract_image_text(path)
            return {
                "summary": "Extracted visual content from image using deterministic local flow",
                "extracted_text": text,
                "source": str(path),
            }

        if suffix in {".txt", ".md", ".markdown", ".rst"}:
            text = self._extract_text_file(path)
            return {
                "summary": "Extracted text from text document using deterministic local flow",
                "extracted_text": text,
                "source": str(path),
            }

        raise ValueError(f"Unsupported file type: {suffix}")

    def _try_remote_document_call(self, path: Path) -> str | None:
        print(f"[QwenVisionTool] Remote Qwen calls are disabled; using deterministic local flow for {path.name}")
        return None

    def _build_prompt(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            extracted_text = self._extract_pdf_text(path)
            return (
                "You are a document understanding assistant. "
                f"The uploaded file is a PDF named '{path.name}'. "
                f"Extracted text from the document is:\n{extracted_text}\n"
                "Please summarize the document contents in 3-5 bullet points."
            )

        image = Image.open(path)
        image.load()
        width, height = image.size
        return (
            "You are a document understanding assistant. "
            f"The uploaded file is an image named '{path.name}' with dimensions {width}x{height}. "
            "Please describe what this document image appears to contain and mention any visible text."
        )

    def _extract_pdf_text(self, path: Path) -> str:
        reader = PdfReader(str(path))
        parts = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(part for part in parts if part).strip()
        return text or "No selectable text was found in the PDF."

    def _extract_image_text(self, path: Path) -> str:
        image = Image.open(path)
        image.load()
        width, height = image.size
        return (
            f"Image detected: {path.name}\n"
            f"Dimensions: {width}x{height}\n"
            f"The Qwen Vision tool would analyze this image for OCR and layout understanding."
        )

    def _extract_text_file(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")


class YoloObjectDetectionTool:
    """A local object detection tool using YOLO.

    Attempts to load the actual YOLO model from the `ultralytics` package.
    If `ultralytics` is not installed or loading fails, it falls back to a
    realistic, deterministic simulation of YOLO object detection.
    """

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            # Lazy load YOLO to avoid startup delays or warnings if not installed
            self.model = YOLO(self.model_name)
            print(f"[YoloObjectDetectionTool] Successfully loaded YOLO model: {self.model_name}")
        except Exception:
            print("[YoloObjectDetectionTool] ultralytics package not available or model load failed. Using simulated object detection.")

    def invoke(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            raise ValueError(f"Unsupported image type for YOLO: {suffix}")

        try:
            image = Image.open(path)
            image.load()
            width, height = image.size
        except Exception as exc:
            raise ValueError(f"Failed to open image file {path.name}: {exc}")

        # If real YOLO model is loaded, run it
        if self.model is not None:
            try:
                results = self.model(str(path))
                detected = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        name = result.names[cls]
                        detected.append({
                            "class": name,
                            "confidence": round(conf, 2),
                            "box": [round(x, 1) for x in xyxy],
                        })
                return {
                    "summary": f"Detected {len(detected)} object(s) in {path.name} using real YOLO.",
                    "detected_objects": detected,
                    "source": str(path),
                }
            except Exception as exc:
                print(f"[YoloObjectDetectionTool] Real YOLO execution failed: {exc}. Falling back to simulation.")

        # Simulated fallback
        detected = self._get_simulated_detections(path.name, width, height)
        summary = f"Detected {len(detected)} object(s) in {path.name} using simulated YOLO."
        return {
            "summary": summary,
            "detected_objects": detected,
            "source": str(path),
        }

    def _get_simulated_detections(self, filename: str, width: int, height: int) -> list[dict[str, Any]]:
        # Provide some realistic object detections based on the file name/metadata
        if "document" in filename.lower():
            return [
                {"class": "document", "confidence": 0.95, "box": [10.0, 10.0, float(width - 10), float(height - 10)]},
                {"class": "text_block", "confidence": 0.88, "box": [40.0, 50.0, 450.0, 150.0]},
            ]
        return [
            {"class": "person", "confidence": 0.92, "box": [50.0, 30.0, 200.0, float(height - 30)]},
            {"class": "laptop", "confidence": 0.89, "box": [180.0, 120.0, 350.0, float(height - 20)]},
            {"class": "chair", "confidence": 0.78, "box": [320.0, 100.0, float(width - 20), float(height - 10)]},
        ]
