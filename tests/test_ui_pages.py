"""Tests for UI helper behavior."""

from __future__ import annotations

from youtube_rag.models.qa import QAResponse, QAStatus
from youtube_rag.models.source import SourceProcessingStatus, SourceRecord, SourceType
from youtube_rag.ui import pages


def test_record_question_attempt_and_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(pages.st, "session_state", {})
    pages._initialize_ui_state()
    monkeypatch.setattr(pages.time, "time", lambda: 100.0)

    pages._record_question_attempt()

    assert pages.st.session_state["ui_metrics"]["questions_asked"] == 1
    assert pages._is_rate_limited(2.0) is True


def test_record_qa_result_updates_counters(monkeypatch) -> None:
    monkeypatch.setattr(pages.st, "session_state", {})
    pages._initialize_ui_state()

    pages._record_qa_result(
        QAResponse(success=True, status=QAStatus.ANSWERED, message="ok", answer="answer")
    )
    pages._record_qa_result(
        QAResponse(success=False, status=QAStatus.NO_CONTEXT, message="none")
    )
    pages._record_qa_result(
        QAResponse(success=False, status=QAStatus.ERROR, message="err")
    )

    metrics = pages.st.session_state["ui_metrics"]
    assert metrics["answers_returned"] == 1
    assert metrics["no_context_answers"] == 1
    assert metrics["qa_errors"] == 1


def test_build_source_options_marks_latest_processed(monkeypatch) -> None:
    monkeypatch.setattr(pages.st, "session_state", {"processed_video_id": "vid123"})
    ready_sources = [
        SourceRecord(
            source_id="youtube:vid123",
            source_type=SourceType.YOUTUBE,
            external_id="vid123",
            title="YouTube Video vid123",
            processing_status=SourceProcessingStatus.READY,
        ),
        SourceRecord(
            source_id="youtube:vid456",
            source_type=SourceType.YOUTUBE,
            external_id="vid456",
            title="YouTube Video vid456",
            processing_status=SourceProcessingStatus.READY,
        ),
    ]

    options, default_selected_labels = pages._build_source_options(ready_sources)

    assert options["Latest Processed: YouTube Video vid123 (youtube)"] == "youtube:vid123"
    assert options["YouTube Video vid456 (youtube)"] == "youtube:vid456"
    assert default_selected_labels == ["Latest Processed: YouTube Video vid123 (youtube)"]
