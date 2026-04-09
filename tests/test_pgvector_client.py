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

    def execute(self, query: str, parameters: tuple[object, ...]) -> FakeResult:
        self.calls.append({"query": query, "parameters": parameters})
        return FakeResult([])


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
    assert parameters == ("[0.1,0.2]", "[0.1,0.2]", 0.75, "vid123", "[0.1,0.2]", 5)
