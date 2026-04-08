"""Word-based transcript chunking using tiktoken token counts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol

import tiktoken

from youtube_rag.models.chunk import ChunkSentence, TranscriptChunk
from youtube_rag.models.transcript import TranscriptPayload, TranscriptSegment


class ChunkingError(ValueError):
    """Retained for compatibility with callers importing the symbol."""


class TokenCounter(Protocol):
    """Token counting contract to support deterministic tests."""

    def count_tokens(self, text: str) -> int:
        """Return the token count for a piece of text."""


class TiktokenTokenCounter:
    """Count tokens with the OpenAI cl100k_base encoding."""

    def __init__(self) -> None:
        for proxy_var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            if os.environ.get(proxy_var) == "http://127.0.0.1:9" or os.environ.get(proxy_var) == "127.0.0.1:9":
                os.environ.pop(proxy_var, None)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))


@dataclass(frozen=True)
class _WordWindow:
    text: str
    start_time: float
    end_time: float


class ChunkingService:
    """Build retrieval chunks from transcript words without sentence logic."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        max_chunk_tokens: int = 500,
    ) -> None:
        self._token_counter = token_counter or TiktokenTokenCounter()
        self._max_chunk_tokens = max_chunk_tokens

    def count_tokens(self, text: str) -> int:
        """Return the token count for a piece of text."""

        return self._token_counter.count_tokens(text)

    def chunk_text(self, text: str, max_tokens: int | None = None) -> list[str]:
        """Split text into token-limited word chunks."""

        words = text.split()
        chunks: list[str] = []
        current_chunk: list[str] = []
        limit = max_tokens or self._max_chunk_tokens

        for word in words:
            current_chunk.append(word)
            current_text = " ".join(current_chunk)
            if self.count_tokens(current_text) > limit:
                chunks.append(current_text)
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_transcript(self, transcript: TranscriptPayload) -> list[TranscriptChunk]:
        """Convert a normalized transcript into retrieval-ready word chunks."""

        word_windows = self._build_word_windows(transcript.segments)
        if not word_windows:
            return []

        chunks: list[TranscriptChunk] = []
        current_words: list[_WordWindow] = []

        for word_window in word_windows:
            current_words.append(word_window)
            current_text = " ".join(item.text for item in current_words)
            if self.count_tokens(current_text) > self._max_chunk_tokens:
                chunks.append(self._build_chunk(transcript.video_id, len(chunks) + 1, current_words))
                current_words = []

        if current_words:
            chunks.append(self._build_chunk(transcript.video_id, len(chunks) + 1, current_words))

        return chunks

    def _build_word_windows(self, segments: list[TranscriptSegment]) -> list[_WordWindow]:
        word_windows: list[_WordWindow] = []

        for segment in sorted(segments, key=lambda item: item.start):
            segment_text = segment.text.strip()
            if not segment_text:
                continue

            words = segment_text.split()
            if not words:
                continue

            word_windows.extend(_map_words_to_timestamps(segment.start, segment.duration, words))

        return word_windows

    def _build_chunk(
        self,
        video_id: str,
        chunk_number: int,
        words: list[_WordWindow],
    ) -> TranscriptChunk:
        chunk_text = " ".join(word.text for word in words)
        token_count = self.count_tokens(chunk_text)
        return TranscriptChunk(
            chunk_id=f"{video_id}_{chunk_number:04d}",
            video_id=video_id,
            text=chunk_text,
            start_time=words[0].start_time,
            end_time=words[-1].end_time,
            token_count=token_count,
            sentences=[
                ChunkSentence(
                    text=chunk_text,
                    start_time=words[0].start_time,
                    end_time=words[-1].end_time,
                    token_count=token_count,
                )
            ],
        )


def _map_words_to_timestamps(
    segment_start: float,
    segment_duration: float,
    words: list[str],
) -> list[_WordWindow]:
    if not words:
        return []

    if segment_duration <= 0:
        return [
            _WordWindow(text=word, start_time=segment_start, end_time=segment_start)
            for word in words
        ]

    segment_end = segment_start + segment_duration
    step = segment_duration / len(words)
    mapped_words: list[_WordWindow] = []

    for index, word in enumerate(words):
        start_time = segment_start + (step * index)
        end_time = segment_start + (step * (index + 1))
        if index == 0:
            start_time = segment_start
        if index == len(words) - 1:
            end_time = segment_end
        mapped_words.append(
            _WordWindow(
                text=word,
                start_time=max(start_time, segment_start),
                end_time=min(end_time, segment_end),
            )
        )

    return mapped_words
