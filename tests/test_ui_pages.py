"""Tests for UI helper behavior."""

from __future__ import annotations

from youtube_rag.models.qa import QAResponse, QAStatus
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
