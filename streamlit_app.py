"""Streamlit entrypoint for the YouTube RAG Pipeline MVP."""

from __future__ import annotations

# ✅ Load environment variables FIRST (after __future__)
from dotenv import load_dotenv
from pathlib import Path
import sys

# 🔥 Force correct .env loading from project root
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ✅ Add src to path
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import streamlit as st

from youtube_rag.config import get_config, get_missing_required_settings
from youtube_rag.services.video_ingestion import (
    InMemoryVideoRegistry,
    StaticAvailabilityChecker,
    VideoIngestionService,
)
from youtube_rag.services.transcript_service import TranscriptService
from youtube_rag.ui.pages import render_video_intake_page
from youtube_rag.utils.logging import configure_logging


def main() -> None:
    """Boot the Streamlit shell and Phase 1-2 flow."""

    missing_settings = get_missing_required_settings()

    # Configure logging
    if missing_settings:
        configure_logging()
    else:
        configure_logging(get_config().log_level)

    # Initialize ingestion service
    ingestion_service = VideoIngestionService(
        duplicate_repository=InMemoryVideoRegistry(),
        availability_checker=StaticAvailabilityChecker(),
    )
    transcript_service = TranscriptService()

    # Render UI
    render_video_intake_page(
        ingestion_service,
        transcript_service,
        missing_settings=missing_settings,
    )


if __name__ == "__main__":
    main()
