"""Transcript extraction and normalization logic."""

from __future__ import annotations

import logging
from typing import Iterable, Protocol

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApiException,
)

from youtube_rag.models.transcript import (
    TranscriptExtractionResponse,
    TranscriptMetadata,
    TranscriptPayload,
    TranscriptProcessingStatus,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)


class TranscriptSnippet(Protocol):
    """Protocol for transcript snippets returned by the fetch client."""

    text: str
    start: float
    duration: float


class FetchedTranscript(Protocol):
    """Protocol for fetched transcript data."""

    video_id: str
    language: str
    language_code: str
    is_generated: bool
    snippets: list[TranscriptSnippet]


class TranscriptClient(Protocol):
    """Transcript fetch client contract to allow mocking in tests."""

    def fetch(
        self,
        video_id: str,
        languages: Iterable[str] = ("en",),
        preserve_formatting: bool = False,
    ) -> FetchedTranscript:
        """Fetch transcript data for a specific video."""


class YouTubeTranscriptClient:
    """Concrete transcript client backed by youtube-transcript-api."""

    def __init__(self, api: YouTubeTranscriptApi | None = None) -> None:
        self._api = api or YouTubeTranscriptApi()

    def fetch(
        self,
        video_id: str,
        languages: Iterable[str] = ("en",),
        preserve_formatting: bool = False,
    ) -> FetchedTranscript:
        return self._api.fetch(
            video_id,
            languages=languages,
            preserve_formatting=preserve_formatting,
        )


class TranscriptService:
    """Fetch and normalize YouTube transcripts for downstream processing."""

    def __init__(
        self,
        transcript_client: TranscriptClient | None = None,
        languages: Iterable[str] = ("en",),
        preserve_formatting: bool = False,
    ) -> None:
        self._transcript_client = transcript_client or YouTubeTranscriptClient()
        self._languages = tuple(languages)
        self._preserve_formatting = preserve_formatting

    def extract(self, video_id: str) -> TranscriptExtractionResponse:
        """Fetch transcript data and normalize it into the internal schema."""

        try:
            fetched_transcript = self._transcript_client.fetch(
                video_id,
                languages=self._languages,
                preserve_formatting=self._preserve_formatting,
            )
        except TranscriptsDisabled:
            logger.info("Transcript extraction failed: transcripts disabled", extra={"video_id": video_id})
            return TranscriptExtractionResponse(
                success=False,
                status=TranscriptProcessingStatus.TRANSCRIPTS_DISABLED,
                message="Transcripts are disabled for this video.",
            )
        except NoTranscriptFound:
            logger.info("Transcript extraction failed: no transcript found", extra={"video_id": video_id})
            return TranscriptExtractionResponse(
                success=False,
                status=TranscriptProcessingStatus.NOT_AVAILABLE,
                message="No usable transcript was found for this video.",
            )
        except VideoUnavailable:
            logger.warning("Transcript extraction failed: video unavailable", extra={"video_id": video_id})
            return TranscriptExtractionResponse(
                success=False,
                status=TranscriptProcessingStatus.NOT_AVAILABLE,
                message="This video is unavailable for transcript extraction.",
            )
        except (CouldNotRetrieveTranscript, YouTubeTranscriptApiException) as exc:
            logger.exception("Transcript extraction failed", extra={"video_id": video_id})
            return TranscriptExtractionResponse(
                success=False,
                status=TranscriptProcessingStatus.EXTRACTION_ERROR,
                message=f"Transcript extraction failed: {exc}",
            )

        segments = self._normalize_segments(fetched_transcript.snippets)
        if not segments:
            logger.info("Transcript extraction returned no usable segments", extra={"video_id": video_id})
            return TranscriptExtractionResponse(
                success=False,
                status=TranscriptProcessingStatus.EMPTY,
                message="The transcript was retrieved but contained no usable text segments.",
            )

        payload = TranscriptPayload(
            video_id=fetched_transcript.video_id,
            segments=segments,
            metadata=TranscriptMetadata(
                language=fetched_transcript.language,
                language_code=fetched_transcript.language_code,
                is_generated=fetched_transcript.is_generated,
                total_segments=len(segments),
                total_duration_seconds=max(segment.start + segment.duration for segment in segments),
            ),
        )
        logger.info(
            "Transcript extraction complete",
            extra={"video_id": video_id, "total_segments": payload.metadata.total_segments},
        )
        return TranscriptExtractionResponse(
            success=True,
            status=TranscriptProcessingStatus.READY,
            message="Transcript extraction complete.",
            payload=payload,
        )

    @staticmethod
    def _normalize_segments(snippets: Iterable[TranscriptSnippet]) -> list[TranscriptSegment]:
        """Normalize transcript snippets and drop empty or malformed entries."""

        normalized_segments: list[TranscriptSegment] = []
        for snippet in snippets:
            text = snippet.text.strip()
            if not text:
                continue

            normalized_segments.append(
                TranscriptSegment(
                    text=text,
                    start=max(float(snippet.start), 0.0),
                    duration=max(float(snippet.duration), 0.0),
                )
            )

        normalized_segments.sort(key=lambda segment: segment.start)
        return normalized_segments
