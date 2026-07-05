import tempfile
import unittest
from pathlib import Path

from app.tools import QwenVisionTool


class QwenVisionToolTest(unittest.TestCase):
    def test_markdown_files_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "README.md"
            path.write_text("# Hello\nThis is a test document.", encoding="utf-8")

            tool = QwenVisionTool()
            result = tool.invoke(str(path))

            self.assertEqual(result["source"], str(path))
            self.assertIn("Hello", result["extracted_text"])
            self.assertIn("test document", result["extracted_text"])


if __name__ == "__main__":
    unittest.main()
