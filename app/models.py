from __future__ import annotations

from typing import Annotated, Any, Literal

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """State passed through the LangGraph workflow."""

    user_request: str
    file_path: str | None
    file_type: str | None
    extracted_text: str
    detected_objects: list[dict[str, Any]] | None
    supervisor_plan: str
    worker_result: str
    final_response: str
    route: Literal["document_extraction", "object_detection"]


class DocumentExtractionResult(TypedDict):
    """Shape returned by the document extraction worker."""

    summary: str
    extracted_text: str
    source: str
