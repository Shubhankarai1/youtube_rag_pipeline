"""Tests for pgvector repository query construction."""

from __future__ import annotations

from contextlib import contextmanager

from youtube_rag.db.pgvector_client import PgVectorChunkRepository


class FakeResult:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict]:
        return self._rows


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.commit_calls = 0

    def execute(self, query: str, parameters: tuple[object, ...]) -> FakeResult:
        self.calls.append({"query": query, "parameters": parameters})
        return FakeResult([])

    def commit(self) -> None:
        self.commit_calls += 1


class FakePgVectorChunkRepository(PgVectorChunkRepository):
    def __init__(self, fake_connection: FakeConnection) -> None:
        super().__init__("postgresql://unused")
        self._fake_connection = fake_connection

    @contextmanager
    def _get_connection(self):
        yield self._fake_connection


def test_retrieve_similar_chunks_applies_similarity_threshold_and_video_filter() -> None:
    connection = FakeConnection()
    repository = FakePgVectorChunkRepository(connection)

    repository.retrieve_similar_chunks(
        [0.1, 0.2],
        top_k=5,
        similarity_threshold=0.75,
        video_id="vid123",
    )

    assert len(connection.calls) == 1
    query = connection.calls[0]["query"]
    parameters = connection.calls[0]["parameters"]

    assert "1 - (embedding <=> %s::vector) >= %s" in query
    assert "video_id = %s" in query
    assert "LEFT JOIN sources ON sources.id = video_chunks.source_id" in query
    assert parameters == ("[0.1,0.2]", "[0.1,0.2]", 0.75, "vid123", "[0.1,0.2]", 5)


def test_register_youtube_source_persists_source_registry_entry() -> None:
    connection = FakeConnection()
    repository = FakePgVectorChunkRepository(connection)

    source = repository.register_youtube_source("vid123")

    assert source.source_id == "youtube:vid123"
    assert len(connection.calls) == 1
    assert "INSERT INTO sources" in connection.calls[0]["query"]
    assert connection.calls[0]["parameters"] == (
        "youtube:vid123",
        "youtube",
        "vid123",
        "YouTube Video vid123",
        "ready",
        "https://www.youtube.com/watch?v=vid123",
        "https://www.youtube.com/watch?v=vid123",
    )
    assert connection.commit_calls == 1
