"""Sentence-aware transcript chunking using BERT-style token counts."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Protocol

from nltk import data as nltk_data
from nltk import download as nltk_download
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

from youtube_rag.models.chunk import ChunkSentence, TranscriptChunk
from youtube_rag.models.transcript import TranscriptPayload, TranscriptSegment


logger = logging.getLogger(__name__)


class ChunkingError(ValueError):
    """Raised when transcript content cannot be chunked safely."""


class TokenCounter(Protocol):
    """Token counting contract to support deterministic tests."""

    def count_tokens(self, text: str) -> int:
        """Return the token count for a piece of text."""


class SentenceSplitter(Protocol):
    """Sentence splitting contract to support deterministic tests."""

    def split(self, text: str) -> list[str]:
        """Return sentence-ordered text spans."""


class BertTokenCounter:
    """Count tokens with the HuggingFace bert-base-uncased tokenizer."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.tokenize(text))


class NLTKSentenceSplitter:
    """Split transcript text into sentences with NLTK."""

    def __init__(self) -> None:
        self._ensure_tokenizer_data()

    def split(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        return [sentence.strip() for sentence in sent_tokenize(stripped) if sentence.strip()]

    def _ensure_tokenizer_data(self) -> None:
        for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            try:
                nltk_data.find(resource)
            except LookupError:
                resource_name = resource.split("/")[-1]
                try:
                    nltk_download(resource_name, quiet=True)
                except Exception:  # pragma: no cover - defensive runtime path
                    logger.warning("Could not download NLTK resource", extra={"resource": resource_name})


@dataclass(frozen=True)
class _SentenceWindow:
    text: str
    start_time: float
    end_time: float
    token_count: int


class ChunkingService:
    """Build retrieval chunks from whole sentences under a BERT token limit."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        sentence_splitter: SentenceSplitter | None = None,
        max_chunk_tokens: int = 500,
    ) -> None:
        self._token_counter = token_counter or BertTokenCounter()
        self._sentence_splitter = sentence_splitter or NLTKSentenceSplitter()
        self._max_chunk_tokens = max_chunk_tokens

    def count_tokens(self, text: str) -> int:
        """Return the token count for a piece of text."""

        return self._token_counter.count_tokens(text)

    def chunk_text(self, text: str, max_tokens: int | None = None) -> list[str]:
        """Split text into sentence-preserving token-limited chunks."""

        limit = max_tokens or self._max_chunk_tokens
        sentences = self._sentence_splitter.split(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current_sentences: list[str] = []

        for sentence in sentences:
            sentence_token_count = self.count_tokens(sentence)
            if sentence_token_count > limit:
                raise ChunkingError(
                    f"Single sentence exceeds chunk token limit: {sentence_token_count} > {limit}"
                )

            candidate_sentences = current_sentences + [sentence]
            candidate_text = " ".join(candidate_sentences)
            if current_sentences and self.count_tokens(candidate_text) > limit:
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence]
                continue

            current_sentences = candidate_sentences

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def chunk_transcript(self, transcript: TranscriptPayload) -> list[TranscriptChunk]:
        """Convert a normalized transcript into sentence-aware retrieval chunks."""

        sentence_windows = self._build_sentence_windows(transcript.segments)
        if not sentence_windows:
            return []

        chunks: list[TranscriptChunk] = []
        current_sentences: list[_SentenceWindow] = []

        for sentence_window in sentence_windows:
            if sentence_window.token_count > self._max_chunk_tokens:
                raise ChunkingError(
                    "Single sentence exceeds chunk token limit and cannot be split safely."
                )

            candidate_sentences = current_sentences + [sentence_window]
            candidate_text = " ".join(sentence.text for sentence in candidate_sentences)
            if current_sentences and self.count_tokens(candidate_text) > self._max_chunk_tokens:
                chunks.append(self._build_chunk(transcript.video_id, len(chunks) + 1, current_sentences))
                current_sentences = [sentence_window]
                continue

            current_sentences = candidate_sentences

        if current_sentences:
            chunks.append(self._build_chunk(transcript.video_id, len(chunks) + 1, current_sentences))

        return chunks

    def _build_sentence_windows(self, segments: list[TranscriptSegment]) -> list[_SentenceWindow]:
        sentence_windows: list[_SentenceWindow] = []

        for segment in sorted(segments, key=lambda item: item.start):
            segment_text = segment.text.strip()
            if not segment_text:
                continue

            sentence_texts = self._sentence_splitter.split(segment_text)
            sentence_windows.extend(
                _map_sentences_to_timestamps(
                    segment_text=segment_text,
                    segment_start=segment.start,
                    segment_duration=segment.duration,
                    sentence_texts=sentence_texts,
                    token_counter=self._token_counter,
                )
            )

        return sentence_windows

    def _build_chunk(
        self,
        video_id: str,
        chunk_number: int,
        sentences: list[_SentenceWindow],
    ) -> TranscriptChunk:
        chunk_text = " ".join(sentence.text for sentence in sentences)
        token_count = self.count_tokens(chunk_text)
        return TranscriptChunk(
            chunk_id=f"{video_id}_{chunk_number:04d}",
            video_id=video_id,
            text=chunk_text,
            start_time=sentences[0].start_time,
            end_time=sentences[-1].end_time,
            token_count=token_count,
            sentences=[
                ChunkSentence(
                    text=sentence.text,
                    start_time=sentence.start_time,
                    end_time=sentence.end_time,
                    token_count=sentence.token_count,
                )
                for sentence in sentences
            ],
        )


def _map_sentences_to_timestamps(
    *,
    segment_text: str,
    segment_start: float,
    segment_duration: float,
    sentence_texts: list[str],
    token_counter: TokenCounter,
) -> list[_SentenceWindow]:
    if not sentence_texts:
        return []

    spans = _resolve_sentence_spans(segment_text, sentence_texts)
    if segment_duration <= 0:
        return [
            _SentenceWindow(
                text=sentence_text,
                start_time=segment_start,
                end_time=segment_start,
                token_count=token_counter.count_tokens(sentence_text),
            )
            for sentence_text in sentence_texts
        ]

    segment_end = segment_start + segment_duration
    text_length = max(len(segment_text), 1)
    sentence_windows: list[_SentenceWindow] = []
    previous_end_time = segment_start

    for index, (sentence_text, span_start, span_end) in enumerate(spans):
        start_ratio = span_start / text_length
        end_ratio = span_end / text_length
        start_time = segment_start + (segment_duration * start_ratio)
        end_time = segment_start + (segment_duration * end_ratio)

        if index == 0:
            start_time = segment_start
        else:
            start_time = previous_end_time
        if index == len(spans) - 1:
            end_time = segment_end

        previous_end_time = end_time
        sentence_windows.append(
            _SentenceWindow(
                text=sentence_text,
                start_time=max(start_time, segment_start),
                end_time=min(end_time, segment_end),
                token_count=token_counter.count_tokens(sentence_text),
            )
        )

    return sentence_windows


def _resolve_sentence_spans(segment_text: str, sentence_texts: list[str]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    cursor = 0

    for sentence_text in sentence_texts:
        search_pattern = re.escape(sentence_text.strip())
        match = re.search(search_pattern, segment_text[cursor:])
        if match is None:
            span_start = cursor
            span_end = min(len(segment_text), cursor + len(sentence_text))
        else:
            span_start = cursor + match.start()
            span_end = cursor + match.end()
        spans.append((sentence_text.strip(), span_start, span_end))
        cursor = span_end

    return spans
