from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from app.agents import LangGraphSupervisorApp


def main() -> None:
    app = LangGraphSupervisorApp()
    sample_path = Path("sample_document.png")

    if not sample_path.exists():
        image = Image.new("RGB", (500, 220), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 480, 200), outline="black", width=3)
        draw.text((40, 60), "Sample Document", fill="black")
        draw.text((40, 95), "LangGraph Supervisor Demo", fill="black")
        draw.text((40, 130), "Uploaded image for extraction", fill="black")
        image.save(sample_path)

    # 1. Document Extraction Flow
    result_extraction = app.invoke(
        user_request="Extract the contents from this uploaded document and summarize it.",
        file_path=str(sample_path),
    )

    print("=" * 60)
    print("Supervisor/Worker LangGraph Demo: Document Extraction")
    print("=" * 60)
    print(result_extraction["final_response"])
    print("=" * 60)
    print()

    # 2. Object Detection Flow
    result_detection = app.invoke(
        user_request="Locate, detect, and list all objects or elements within this image.",
        file_path=str(sample_path),
    )

    print("=" * 60)
    print("Supervisor/Worker LangGraph Demo: Object Detection")
    print("=" * 60)
    print(result_detection["final_response"])
    print("=" * 60)


if __name__ == "__main__":
    main()
