import tempfile
import unittest
from pathlib import Path
from PIL import Image

from app.tools import YoloObjectDetectionTool
from app.agents import (
    SupervisorAgent,
    ObjectDetectionWorker,
    LangGraphSupervisorApp,
)
from app.models import AgentState


class TestAgents(unittest.TestCase):
    def setUp(self) -> None:
        self.app = LangGraphSupervisorApp()

    def test_yolo_tool_simulated(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_image.png"
            image = Image.new("RGB", (300, 200), "white")
            image.save(img_path)

            tool = YoloObjectDetectionTool()
            res = tool.invoke(str(img_path))
            self.assertEqual(res["source"], str(img_path))
            self.assertTrue(len(res["detected_objects"]) > 0)
            self.assertIn("person", [obj["class"] for obj in res["detected_objects"]])

            # Test document named image
            doc_img_path = Path(tmpdir) / "sample_document.png"
            image.save(doc_img_path)
            res_doc = tool.invoke(str(doc_img_path))
            self.assertIn("document", [obj["class"] for obj in res_doc["detected_objects"]])

    def test_supervisor_agent_routing_fallback(self) -> None:
        supervisor = SupervisorAgent()
        supervisor.client = None
        
        # Test routing for object detection
        state_det: AgentState = {
            "user_request": "Please detect objects in this picture.",
            "file_path": "test.png",
            "file_type": ".png",
        }
        res_det = supervisor.route_request(state_det)
        self.assertEqual(res_det["route"], "object_detection")

        # Test routing for document extraction
        state_doc: AgentState = {
            "user_request": "Please extract the text and summarize.",
            "file_path": "test.png",
            "file_type": ".png",
        }
        res_doc = supervisor.route_request(state_doc)
        self.assertEqual(res_doc["route"], "document_extraction")

    def test_supervisor_agent_consolidation_fallback(self) -> None:
        supervisor = SupervisorAgent()
        supervisor.client = None
        
        # With object detections
        state: AgentState = {
            "user_request": "Find objects.",
            "worker_result": "Detected 2 objects.",
            "detected_objects": [
                {"class": "person", "confidence": 0.9, "box": [0, 0, 10, 10]},
                {"class": "person", "confidence": 0.8, "box": [10, 10, 20, 20]},
                {"class": "laptop", "confidence": 0.95, "box": [0, 0, 5, 5]},
            ]
        }
        res = supervisor.consolidate_response(state)
        self.assertIn("2 person(s)", res["final_response"])
        self.assertIn("1 laptop(s)", res["final_response"])

    def test_full_app_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test_image.png"
            image = Image.new("RGB", (300, 200), "white")
            image.save(img_path)

            # Test object detection workflow
            result = self.app.invoke(
                user_request="Detect objects in this image.",
                file_path=str(img_path),
            )
            self.assertEqual(result["route"], "object_detection")
            self.assertIn("Supervisor plan:", result["final_response"])
            self.assertIn("Consolidated Response:", result["final_response"])
            self.assertTrue(len(result["detected_objects"]) > 0)


if __name__ == "__main__":
    unittest.main()
