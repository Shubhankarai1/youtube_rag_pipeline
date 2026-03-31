"""YouTube URL parsing and helper utilities."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from youtube_rag.models.video import VideoIntakePayload

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
}
YOUTU_BE_HOSTS = {"youtu.be", "www.youtu.be"}


def extract_video_id(url: str) -> str | None:
    """Extract a YouTube video id from a supported URL."""

    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")

    candidate: str | None = None
    if host in YOUTU_BE_HOSTS:
        candidate = path.split("/", maxsplit=1)[0]
    elif host in YOUTUBE_HOSTS:
        query_video_id = parse_qs(parsed.query).get("v")
        if query_video_id:
            candidate = query_video_id[0]
        elif path.startswith(("embed/", "shorts/", "live/")):
            candidate = path.split("/", maxsplit=1)[1].split("/", maxsplit=1)[0]

    if candidate and _is_valid_video_id(candidate):
        return candidate
    return None


def is_valid_youtube_url(url: str) -> bool:
    """Return True when the URL is a supported YouTube video URL."""

    return extract_video_id(url) is not None


def normalize_youtube_url(video_id: str) -> str:
    """Return the canonical watch URL for a given video id."""

    return f"https://www.youtube.com/watch?v={video_id}"


def build_intake_payload(url: str) -> VideoIntakePayload:
    """Convert a raw user URL into the normalized payload for downstream phases."""

    video_id = extract_video_id(url)
    if video_id is None:
        raise ValueError("A valid YouTube video URL is required.")

    return VideoIntakePayload(
        video_id=video_id,
        source_url=url.strip(),
        normalized_url=normalize_youtube_url(video_id),
    )


def _is_valid_video_id(candidate: str) -> bool:
    return len(candidate) == 11 and all(
        character.isalnum() or character in {"-", "_"} for character in candidate
    )
