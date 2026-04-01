"""Sentence-aware transcript chunking logic using NLTK and BERT tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from youtube_rag.models.chunk import ChunkSentence, TranscriptChunk
from youtube_rag.models.transcript import TranscriptPayload, TranscriptSegment


class ChunkingError(ValueError):
    """Raised when transcript chunking cannot satisfy the configured constraints."""


class TokenCounter(Protocol):
    """Token counting contract to support deterministic tests."""

    def count_tokens(self, text: str) -> int:
        """Return the token count for a piece of text."""


class BertTokenCounter:
    """Count tokens with a HuggingFace BERT tokenizer."""

    def __init__(self, tokenizer_name: str = "bert-base-uncased") -> None:
        self._tokenizer = _load_tokenizer(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


@dataclass(frozen=True)
class _SentenceWindow:
    text: str
    start_time: float
    end_time: float
    token_count: int


class ChunkingService:
    """Build retrieval chunks from transcript sentences without splitting sentences."""

    def __init__(
        self,
        sentence_splitter: Callable[[str], list[str]] | None = None,
        token_counter: TokenCounter | None = None,
        tokenizer_name: str = "bert-base-uncased",
        max_chunk_tokens: int = 512,
    ) -> None:
        self._sentence_splitter = sentence_splitter or _nltk_sentence_splitter
        self._token_counter = token_counter or BertTokenCounter(tokenizer_name)
        self._max_chunk_tokens = max_chunk_tokens

    def chunk_transcript(self, transcript: TranscriptPayload) -> list[TranscriptChunk]:
        """Convert a normalized transcript into retrieval-ready sentence chunks."""

        sentence_windows = self._build_sentence_windows(transcript.segments)
        if not sentence_windows:
            return []

        chunks: list[TranscriptChunk] = []
        current_sentences: list[_SentenceWindow] = []
        current_token_count = 0

        for sentence in sentence_windows:
            if sentence.token_count > self._max_chunk_tokens:
                raise ChunkingError(
                    "A single sentence exceeds the configured BERT token limit and cannot be chunked "
                    "without splitting the sentence."
                )

            if current_sentences and current_token_count + sentence.token_count > self._max_chunk_tokens:
                chunks.append(self._build_chunk(transcript.video_id, len(chunks) + 1, current_sentences))
                current_sentences = []
                current_token_count = 0

            current_sentences.append(sentence)
            current_token_count += sentence.token_count

        if current_sentences:
            chunks.append(self._build_chunk(transcript.video_id, len(chunks) + 1, current_sentences))

        return chunks

    def _build_sentence_windows(self, segments: list[TranscriptSegment]) -> list[_SentenceWindow]:
        sentence_windows: list[_SentenceWindow] = []

        for segment in sorted(segments, key=lambda item: item.start):
            segment_text = segment.text.strip()
            if not segment_text:
                continue

            sentences = [sentence.strip() for sentence in self._sentence_splitter(segment_text) if sentence.strip()]
            if not sentences:
                sentences = [segment_text]

            for sentence_text, start_time, end_time in _map_sentences_to_timestamps(
                segment_text,
                segment.start,
                segment.duration,
                sentences,
            ):
                sentence_windows.append(
                    _SentenceWindow(
                        text=sentence_text,
                        start_time=start_time,
                        end_time=end_time,
                        token_count=self._token_counter.count_tokens(sentence_text),
                    )
                )

        return sentence_windows

    def _build_chunk(
        self,
        video_id: str,
        chunk_number: int,
        sentences: list[_SentenceWindow],
    ) -> TranscriptChunk:
        chunk_sentences = [
            ChunkSentence(
                text=sentence.text,
                start_time=sentence.start_time,
                end_time=sentence.end_time,
                token_count=sentence.token_count,
            )
            for sentence in sentences
        ]
        return TranscriptChunk(
            chunk_id=f"{video_id}_{chunk_number:04d}",
            video_id=video_id,
            text=" ".join(sentence.text for sentence in sentences),
            start_time=sentences[0].start_time,
            end_time=sentences[-1].end_time,
            token_count=sum(sentence.token_count for sentence in sentences),
            sentences=chunk_sentences,
        )


def _nltk_sentence_splitter(text: str) -> list[str]:
    _ensure_nltk_sentence_resources()
    return sent_tokenize(text)


def _ensure_nltk_sentence_resources() -> None:
    for resource_path, download_name in (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab/english", "punkt_tab"),
    ):
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


def _load_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _map_sentences_to_timestamps(
    segment_text: str,
    segment_start: float,
    segment_duration: float,
    sentences: list[str],
) -> list[tuple[str, float, float]]:
    if not sentences:
        return []

    if segment_duration <= 0:
        return [(sentence, segment_start, segment_start) for sentence in sentences]

    mapped_sentences: list[tuple[str, float, float]] = []
    cursor = 0
    total_length = max(len(segment_text), 1)
    segment_end = segment_start + segment_duration

    for index, sentence in enumerate(sentences):
        sentence_index = segment_text.find(sentence, cursor)
        if sentence_index == -1:
            sentence_index = cursor

        sentence_end_index = sentence_index + len(sentence)
        start_ratio = sentence_index / total_length
        end_ratio = sentence_end_index / total_length

        start_time = segment_start + (segment_duration * start_ratio)
        end_time = segment_start + (segment_duration * end_ratio)

        if index == 0:
            start_time = segment_start
        if index == len(sentences) - 1:
            end_time = segment_end

        mapped_sentences.append((sentence, max(start_time, segment_start), min(end_time, segment_end)))
        cursor = sentence_end_index

    return mapped_sentences
