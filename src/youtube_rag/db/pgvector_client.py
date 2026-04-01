"""PostgreSQL and pgvector connection helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from psycopg import connect
from psycopg.rows import dict_row

from youtube_rag.models.chunk import EmbeddedChunk, RetrievedChunk


class PgVectorChunkRepository:
    """Persist and retrieve transcript chunks in PostgreSQL + pgvector."""

    def __init__(self, database_url: str, schema_path: str | Path | None = None) -> None:
        self._database_url = database_url
        self._schema_path = Path(schema_path) if schema_path else Path(__file__).with_name("schema.sql")

    def initialize_schema(self) -> None:
        with self._get_connection() as connection:
            connection.execute(self._schema_path.read_text(encoding="utf-8"))
            connection.commit()

    def has_video(self, video_id: str) -> bool:
        with self._get_connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM video_chunks WHERE video_id = %s LIMIT 1",
                (video_id,),
            ).fetchone()
        return row is not None

    def store_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        with self._get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO video_chunks (
                    id,
                    video_id,
                    chunk_id,
                    content,
                    embedding,
                    start_time,
                    end_time
                )
                VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time
                """,
                [
                    (
                        str(uuid4()),
                        chunk.video_id,
                        chunk.chunk_id,
                        chunk.text,
                        _embedding_to_vector_literal(chunk.embedding),
                        chunk.start_time,
                        chunk.end_time,
                    )
                    for chunk in embedded_chunks
                ],
            )
            connection.commit()

    def retrieve_similar_chunks(
        self,
        query_embedding: list[float],
        *,
        top_k: int,
        similarity_threshold: float,
        video_id: str | None = None,
    ) -> list[RetrievedChunk]:
        where_clause = ""
        vector_literal = _embedding_to_vector_literal(query_embedding)
        parameters: tuple[object, ...]
        if video_id:
            where_clause = "WHERE video_id = %s"
            parameters = (vector_literal, video_id, vector_literal, top_k)
        else:
            parameters = (vector_literal, vector_literal, top_k)

        with self._get_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    chunk_id,
                    video_id,
                    content,
                    start_time,
                    end_time,
                    1 - (embedding <=> %s::vector) AS similarity_score
                FROM video_chunks
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                parameters,
            ).fetchall()

        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                video_id=row["video_id"],
                text=row["content"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                similarity_score=row["similarity_score"],
            )
            for row in rows
            if row["similarity_score"] >= similarity_threshold
        ]

    @contextmanager
    def _get_connection(self) -> Iterator:
        connection = connect(self._database_url, row_factory=dict_row)
        try:
            yield connection
        finally:
            connection.close()


def _embedding_to_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"
