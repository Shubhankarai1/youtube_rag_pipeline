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
from youtube_rag.db.pgvector_client import PgVectorChunkRepository
from youtube_rag.services.chunking_service import ChunkingService
from youtube_rag.services.embedding_service import EmbeddingService, NullEmbeddingService, OpenAIEmbeddingClient
from youtube_rag.services.qa_service import NullQAService, OpenAIAnswerGenerator, QAService
from youtube_rag.services.retrieval_service import RetrievalService
from youtube_rag.services.video_ingestion import (
    InMemoryVideoRegistry,
    StaticAvailabilityChecker,
    VideoIngestionService,
)
from youtube_rag.services.transcript_service import TranscriptService
from youtube_rag.ui.pages import render_video_intake_page
from youtube_rag.utils.logging import configure_logging


def main() -> None:
    """Boot the Streamlit shell and Phase 1-5 flow."""

    missing_settings = get_missing_required_settings()
    config = get_config() if not missing_settings else None

    # Configure logging
    if missing_settings:
        configure_logging()
    else:
        assert config is not None
        configure_logging(config.log_level)

    # Initialize ingestion service
    ingestion_service = VideoIngestionService(
        duplicate_repository=InMemoryVideoRegistry(),
        availability_checker=StaticAvailabilityChecker(),
    )
    transcript_service = TranscriptService()
    chunking_service = ChunkingService(
        max_chunk_tokens=config.chunk_max_tokens if config else 500,
    )
    if config is not None:
        repository = PgVectorChunkRepository(config.database_url)
        embedding_client = OpenAIEmbeddingClient(
            api_key=config.openai_api_key,
            model=config.openai_embedding_model,
        )
        embedding_service = EmbeddingService(
            embedding_client=embedding_client,
            repository=repository,
        )
        qa_service = QAService(
            retrieval_service=RetrievalService(
                embedding_client=embedding_client,
                retriever=repository,
                top_k=config.top_k_results,
                similarity_threshold=config.similarity_threshold,
            ),
            answer_generator=OpenAIAnswerGenerator(
                api_key=config.openai_api_key,
                model=config.openai_chat_model,
            ),
        )
    else:
        embedding_service = NullEmbeddingService()
        qa_service = NullQAService()

    # Render UI
    render_video_intake_page(
        ingestion_service,
        transcript_service,
        chunking_service,
        embedding_service,
        qa_service,
        missing_settings=missing_settings,
    )


if __name__ == "__main__":
    main()
