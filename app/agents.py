from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.models import AgentState
from app.tools import QwenVisionTool, YoloObjectDetectionTool

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


class SupervisorAgent:
    """Uses an OpenAI LLM to decide which worker should handle the request."""

    def __init__(
        self,
        doc_worker: DocumentExtractionWorker | None = None,
        det_worker: ObjectDetectionWorker | None = None,
    ) -> None:
        self.doc_worker = doc_worker
        self.det_worker = det_worker
        self.client = None
        self.model_name = os.getenv("SUPERVISOR_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)

    def route_request(self, state: AgentState) -> dict[str, Any]:
        request = state.get("user_request", "")
        file_path = state.get("file_path")
        file_type = state.get("file_type") or (Path(file_path).suffix.lower() if file_path else "")
        state["file_type"] = file_type

        try:
            if not self.client:
                raise RuntimeError("OpenAI API key is not configured")

            print(f"[Supervisor] Using OpenAI LLM {self.model_name} to route the request")
            prompt = (
                "You are a supervisor agent routing work to specialized workers. "
                "Choose exactly one worker from this list: document_extraction, object_detection. "
                "Return a JSON object with keys 'route' and 'reason'.\n"
                f"User request: {request}\n"
                f"Uploaded file type: {file_type or 'none'}\n"
                "If the request is about extracting, reading, summarizing, or understanding text/content in a document/PDF, "
                "route to document_extraction.\n"
                "If the request is about identifying, locating, counting, or detecting objects/things/people/items/obstacles/boxes in an image, "
                "route to object_detection."
            )
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                timeout=120,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a routing supervisor. Respond with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = completion.choices[0].message.content or "{}"
            print(f"[Supervisor] OpenAI LLM response: {content}")
            parsed = self._parse_json(content)
            route = parsed.get("route", "document_extraction")
            reason = parsed.get("reason", "Selected by LLM routing.")
            if route not in {"document_extraction", "object_detection"}:
                route = "document_extraction"
            state["route"] = route
            state["supervisor_plan"] = reason
            return state
        except Exception as exc:
            print(f"[Supervisor] OpenAI LLM routing failed: {exc}")
            pass

        print("[Supervisor] Falling back to deterministic routing")
        request_lower = request.lower()
        if any(keyword in request_lower for keyword in ["detect", "object", "yolo", "locate", "count", "find", "identify"]):
            state["route"] = "object_detection"
            state["supervisor_plan"] = "Fallback routing based on request keywords indicating object detection."
            return state

        if any(keyword in request_lower for keyword in ["extract", "summarize", "read", "document", "image", "pdf"]):
            state["route"] = "document_extraction"
            state["supervisor_plan"] = "Fallback routing based on request keywords indicating document extraction."
            return state

        state["route"] = "document_extraction"
        state["supervisor_plan"] = "Default fallback routing to the document extraction worker."
        return state

    def determine_next_node(self, state: AgentState) -> str:
        return state.get("route") or "document_extraction"

    def consolidate_response(self, state: AgentState) -> dict[str, Any]:
        request = state.get("user_request", "")
        worker_result = state.get("worker_result", "")
        detected_objects = state.get("detected_objects")

        try:
            if not self.client:
                raise RuntimeError("OpenAI API key is not configured")

            print(f"[Supervisor] Using OpenAI LLM {self.model_name} to consolidate the response")
            prompt = (
                "You are a supervisor agent consolidating findings from a specialized worker.\n"
                f"Original User Request: {request}\n"
                f"Worker Result:\n{worker_result}\n\n"
                "Please synthesize a clear, helpful, and concise final response to the user that directly answers their request using the worker's findings."
            )
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.3,
                timeout=120,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Synthesize the worker's findings into a final response.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            consolidated = completion.choices[0].message.content or ""
            state["final_response"] = consolidated.strip()
            return state
        except Exception as exc:
            print(f"[Supervisor] OpenAI LLM consolidation failed: {exc}")
            pass

        print("[Supervisor] Falling back to deterministic consolidation")
        if detected_objects:
            obj_counts = {}
            for obj in detected_objects:
                cls = obj["class"]
                obj_counts[cls] = obj_counts.get(cls, 0) + 1
            counts_str = ", ".join(f"{count} {cls}(s)" for cls, count in obj_counts.items())
            fallback_response = (
                f"Based on the object detection analysis, I found the following objects: {counts_str}.\n"
                f"Details:\n{worker_result}"
            )
        else:
            fallback_response = (
                f"Based on the analysis, here are the findings:\n{worker_result}"
            )

        state["final_response"] = fallback_response
        return state

    def _parse_json(self, content: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


class DocumentExtractionWorker:
    """Extracts structured information from a document using a Qwen Vision tool."""

    def __init__(self, tool: QwenVisionTool) -> None:
        self.tool = tool

    def extract(self, state: AgentState) -> dict[str, Any]:
        file_path = state.get("file_path")
        if not file_path:
            state["worker_result"] = "No file was supplied."
            return state

        result = self.tool.invoke(file_path)
        state["extracted_text"] = result["extracted_text"]
        state["worker_result"] = (
            f"{result['summary']}\n\n{result['extracted_text']}"
        )
        return state


class ObjectDetectionWorker:
    """Detects objects in an image using a YOLO model."""

    def __init__(self, tool: YoloObjectDetectionTool) -> None:
        self.tool = tool

    def detect(self, state: AgentState) -> dict[str, Any]:
        file_path = state.get("file_path")
        if not file_path:
            state["worker_result"] = "No file was supplied for object detection."
            state["detected_objects"] = []
            return state

        try:
            result = self.tool.invoke(file_path)
            state["detected_objects"] = result["detected_objects"]
            details = "\n".join(
                f"- {obj['class']} (confidence: {obj['confidence']:.2f}) at box {obj['box']}"
                for obj in result["detected_objects"]
            )
            state["worker_result"] = f"{result['summary']}\n\n{details}"
        except Exception as exc:
            state["worker_result"] = f"Object detection failed: {exc}"
            state["detected_objects"] = []

        return state


class LangGraphSupervisorApp:
    """Builds the supervisor-worker workflow."""

    def __init__(self) -> None:
        self.doc_worker = DocumentExtractionWorker(QwenVisionTool())
        self.det_worker = ObjectDetectionWorker(YoloObjectDetectionTool())
        self.supervisor = SupervisorAgent(self.doc_worker, self.det_worker)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("supervisor_router", self.supervisor.route_request)
        workflow.add_node("document_extraction_worker", self.doc_worker.extract)
        workflow.add_node("object_detection_worker", self.det_worker.detect)
        workflow.add_node("supervisor_consolidator", self.supervisor.consolidate_response)

        workflow.add_edge(START, "supervisor_router")
        workflow.add_conditional_edges(
            "supervisor_router",
            self.supervisor.determine_next_node,
            {
                "document_extraction": "document_extraction_worker",
                "object_detection": "object_detection_worker",
            },
        )
        workflow.add_edge("document_extraction_worker", "supervisor_consolidator")
        workflow.add_edge("object_detection_worker", "supervisor_consolidator")
        workflow.add_edge("supervisor_consolidator", END)

        return workflow.compile()

    def invoke(self, user_request: str, file_path: str | None = None) -> dict[str, Any]:
        initial_state: AgentState = {
            "user_request": user_request,
            "file_path": file_path,
            "file_type": None,
            "extracted_text": "",
            "detected_objects": [],
            "supervisor_plan": "",
            "worker_result": "",
            "final_response": "",
        }
        result = self.graph.invoke(initial_state)
        if not result.get("final_response"):
            result["final_response"] = self._format_response(result)
        else:
            plan_str = f"Supervisor plan: {result.get('supervisor_plan', 'None')}\n\n"
            if not result["final_response"].startswith("Supervisor plan:"):
                result["final_response"] = f"{plan_str}Consolidated Response:\n{result['final_response']}"
        return result

    def _format_response(self, result: dict[str, Any]) -> str:
        return (
            f"Supervisor plan: {result.get('supervisor_plan', 'None')}\n\n"
            f"Worker output:\n{result.get('worker_result', 'No output')}"
        )
