"""PostgreSQL and pgvector connection helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from psycopg import connect
from psycopg.rows import dict_row

from youtube_rag.models.chunk import EmbeddedChunk, RetrievedChunk
from youtube_rag.models.source import SourceProcessingStatus, SourceRecord, SourceType


class PgVectorChunkRepository:
    """Persist and retrieve transcript chunks in PostgreSQL + pgvector."""

    def __init__(self, database_url: str, schema_path: str | Path | None = None) -> None:
        self._database_url = database_url
        self._schema_path = Path(schema_path) if schema_path else Path(__file__).with_name("schema.sql")

    def initialize_schema(self) -> None:
        with self._get_connection() as connection:
            connection.execute(self._schema_path.read_text(encoding="utf-8"))
            connection.execute("ALTER TABLE video_chunks ADD COLUMN IF NOT EXISTS source_id TEXT")
            connection.execute("ALTER TABLE sources ADD COLUMN IF NOT EXISTS source_url TEXT")
            connection.execute("ALTER TABLE sources ADD COLUMN IF NOT EXISTS normalized_url TEXT")
            self._backfill_sources(connection)
            connection.commit()

    def has_video(self, video_id: str) -> bool:
        with self._get_connection() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM sources
                WHERE source_type = %s
                  AND external_id = %s
                  AND processing_status = %s
                LIMIT 1
                """,
                (SourceType.YOUTUBE.value, video_id, SourceProcessingStatus.READY.value),
            ).fetchone()
        return row is not None

    def get_status(self, video_id: str) -> SourceProcessingStatus | None:
        with self._get_connection() as connection:
            row = connection.execute(
                """
                SELECT processing_status
                FROM sources
                WHERE source_type = %s AND external_id = %s
                LIMIT 1
                """,
                (SourceType.YOUTUBE.value, video_id),
            ).fetchone()
        if row is None:
            return None
        return SourceProcessingStatus(row["processing_status"])

    def start_processing(self, *, video_id: str, source_url: str, normalized_url: str) -> None:
        source = SourceRecord(
            source_id=self._resolve_source_id(video_id),
            source_type=SourceType.YOUTUBE,
            external_id=video_id,
            title=f"YouTube Video {video_id}",
            processing_status=SourceProcessingStatus.PROCESSING,
            source_url=source_url,
            normalized_url=normalized_url,
        )
        with self._get_connection() as connection:
            connection.execute(
                """
                INSERT INTO sources (
                    id,
                    source_type,
                    external_id,
                    title,
                    processing_status,
                    source_url,
                    normalized_url
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_type, external_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    processing_status = EXCLUDED.processing_status,
                    source_url = EXCLUDED.source_url,
                    normalized_url = EXCLUDED.normalized_url
                """,
                (
                    source.source_id,
                    source.source_type.value,
                    source.external_id,
                    source.title,
                    source.processing_status.value,
                    source.source_url,
                    source.normalized_url,
                ),
            )
            connection.commit()

    def mark_ready(self, video_id: str) -> None:
        self._mark_source_status(video_id, SourceProcessingStatus.READY)

    def mark_failed(self, video_id: str) -> None:
        self._mark_source_status(video_id, SourceProcessingStatus.FAILED)

    def ensure_youtube_source(self, video_id: str) -> SourceRecord:
        source = SourceRecord(
            source_id=self._resolve_source_id(video_id),
            source_type=SourceType.YOUTUBE,
            external_id=video_id,
            title=f"YouTube Video {video_id}",
            processing_status=self.get_status(video_id) or SourceProcessingStatus.PROCESSING,
            source_url=f"https://www.youtube.com/watch?v={video_id}",
            normalized_url=f"https://www.youtube.com/watch?v={video_id}",
        )
        with self._get_connection() as connection:
            connection.execute(
                """
                INSERT INTO sources (
                    id,
                    source_type,
                    external_id,
                    title,
                    processing_status,
                    source_url,
                    normalized_url
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_type, external_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_url = EXCLUDED.source_url,
                    normalized_url = EXCLUDED.normalized_url
                """,
                (
                    source.source_id,
                    source.source_type.value,
                    source.external_id,
                    source.title,
                    source.processing_status.value,
                    source.source_url,
                    source.normalized_url,
                ),
            )
            connection.commit()
        return source

    def list_ready_sources(self) -> list[SourceRecord]:
        with self._get_connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    id,
                    source_type,
                    external_id,
                    title,
                    processing_status,
                    source_url,
                    normalized_url
                FROM sources
                WHERE processing_status = %s
                ORDER BY created_at DESC, title ASC
                """,
                (SourceProcessingStatus.READY.value,),
            ).fetchall()

        return [
            SourceRecord(
                source_id=str(row["id"]),
                source_type=SourceType(row["source_type"]),
                external_id=row["external_id"],
                title=row["title"],
                processing_status=SourceProcessingStatus(row["processing_status"]),
                source_url=row["source_url"],
                normalized_url=row["normalized_url"],
            )
            for row in rows
        ]

    def store_embeddings(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        with self._get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO video_chunks (
                        id,
                        video_id,
                        source_id,
                        chunk_id,
                        content,
                        embedding,
                        start_time,
                        end_time
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        source_id = EXCLUDED.source_id,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        start_time = EXCLUDED.start_time,
                        end_time = EXCLUDED.end_time
                    """,
                    [
                        (
                            str(uuid4()),
                            chunk.video_id,
                            chunk.source_id,
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
        source_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        vector_literal = _embedding_to_vector_literal(query_embedding)
        parameters: list[object] = [vector_literal]
        where_conditions = ["1 - (embedding <=> %s::vector) >= %s"]
        parameters.extend([vector_literal, similarity_threshold])

        if video_id:
            where_conditions.append("video_id = %s")
            parameters.append(video_id)
        if source_ids:
            placeholders = ", ".join(["%s"] * len(source_ids))
            where_conditions.append(f"video_chunks.source_id IN ({placeholders})")
            parameters.extend(source_ids)

        parameters.extend([vector_literal, top_k])
        where_clause = "WHERE " + " AND ".join(where_conditions)

        with self._get_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    chunk_id,
                    video_id,
                    video_chunks.source_id,
                    sources.source_type,
                    sources.title AS source_title,
                    content,
                    start_time,
                    end_time,
                    1 - (embedding <=> %s::vector) AS similarity_score
                FROM video_chunks
                LEFT JOIN sources ON sources.id = video_chunks.source_id
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                tuple(parameters),
            ).fetchall()

        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                video_id=row["video_id"],
                source_id=row["source_id"],
                source_type=row["source_type"],
                source_title=row["source_title"],
                text=row["content"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                similarity_score=row["similarity_score"],
            )
            for row in rows
        ]

    def _backfill_sources(self, connection) -> None:
        rows = connection.execute(
            "SELECT DISTINCT video_id FROM video_chunks WHERE video_id IS NOT NULL AND video_id <> ''"
        ).fetchall()
        for row in rows:
            video_id = row["video_id"]
            source_id = self._resolve_source_id(video_id, connection=connection)
            connection.execute(
                """
                INSERT INTO sources (
                    id,
                    source_type,
                    external_id,
                    title,
                    processing_status,
                    source_url,
                    normalized_url
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_type, external_id) DO NOTHING
                """,
                (
                    source_id,
                    SourceType.YOUTUBE.value,
                    video_id,
                    f"YouTube Video {video_id}",
                    SourceProcessingStatus.READY.value,
                    f"https://www.youtube.com/watch?v={video_id}",
                    f"https://www.youtube.com/watch?v={video_id}",
                ),
            )
            connection.execute(
                """
                UPDATE video_chunks
                SET source_id = %s
                WHERE video_id = %s AND source_id IS NULL
                """,
                (source_id, video_id),
            )

    def _resolve_source_id(self, video_id: str, *, connection=None) -> str:
        if connection is None:
            with self._get_connection() as owned_connection:
                return self._resolve_source_id(video_id, connection=owned_connection)

        row = connection.execute(
            """
            SELECT id
            FROM sources
            WHERE source_type = %s AND external_id = %s
            LIMIT 1
            """,
            (SourceType.YOUTUBE.value, video_id),
        ).fetchone()
        if row is not None and row.get("id") is not None:
            return str(row["id"])
        return str(uuid4())

    def _mark_source_status(self, video_id: str, status: SourceProcessingStatus) -> None:
        with self._get_connection() as connection:
            connection.execute(
                """
                UPDATE sources
                SET processing_status = %s
                WHERE source_type = %s AND external_id = %s
                """,
                (status.value, SourceType.YOUTUBE.value, video_id),
            )
            connection.commit()

    @contextmanager
    def _get_connection(self) -> Iterator:
        connection = connect(self._database_url, row_factory=dict_row)
        try:
            yield connection
        finally:
            connection.close()


def _embedding_to_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"
