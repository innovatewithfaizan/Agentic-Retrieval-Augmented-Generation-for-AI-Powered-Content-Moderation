"""Helpers for the Streamlit demo (app.py), kept here so they can be tested
without a browser.

The visitor brings their own API key. It lives only in the Backend built for
one run and is never written to disk, logged, cached or read from the
environment, unless the developer sets SUBRATING_DEV_ENV_KEYS=1 locally.
"""

import io
import json
import os
import uuid

from . import config, labels, llm
from .index import FilmContext, Store
from .reviews import read_matches
from .subtitles import Line, parse_srt, parse_xml

DEV_ENV_KEYS = os.environ.get("SUBRATING_DEV_ENV_KEYS") == "1"
MAX_UPLOAD_LINES = 6000  # the 75 dataset films have 128-3505 lines (median 1278)
DEMO_MODEL = "gpt-5-mini"  # whose committed ratings the app shows without a key


def resolve_key(provider: str, typed: str, dev: bool = DEV_ENV_KEYS) -> str:
    """The key to use: what the visitor typed, or the env key in dev mode only."""
    typed = (typed or "").strip()
    if typed or not dev:
        return typed
    return os.environ.get(llm.PROVIDERS[provider]["env"], "")


def make_backend(provider: str, model: str, key: str, effort: str) -> llm.Backend:
    if not key:  # Backend would fall back to the environment; never do that here
        raise ValueError("an API key is required")
    return llm.Backend(provider, model, api_key=key, effort=effort)


def films() -> list[dict]:
    return read_matches()


def parse_upload(name: str, data: bytes) -> list[Line]:
    """Parse an uploaded .srt or OpenSubtitles .xml file."""
    if name.lower().endswith(".xml"):
        lines = parse_xml(io.BytesIO(data))
    elif name.lower().endswith(".srt"):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
        lines = parse_srt(text)
    else:
        raise ValueError("upload an .srt or OpenSubtitles .xml file")
    if not lines:
        raise ValueError("no subtitle lines found in that file")
    if len(lines) > MAX_UPLOAD_LINES:
        raise ValueError(f"file has {len(lines)} lines; the demo accepts up to {MAX_UPLOAD_LINES}")
    return lines


def dataset_context(film: dict, store: Store) -> FilmContext:
    return FilmContext.from_dataset(film, store)


def upload_context(lines: list[Line], store: Store) -> FilmContext:
    """A context under a random id, so uploads never collide or show up for
    another visitor. Call forget() when done."""
    film_id = f"upload-{uuid.uuid4().hex}"
    store.ensure(film_id, lines)
    return FilmContext(film_id, lines, store)


def forget(ctx: FilmContext) -> None:
    with ctx.store._lock:
        ctx.store.col.delete(where={"film_id": ctx.film_id})


def precomputed(film_id: str, mode: str, model: str = DEMO_MODEL) -> dict | None:
    path = config.ratings_dir(model, mode) / f"{film_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def review_labels(film_id: str) -> dict | None:
    rec = labels.read_labels().get(film_id)
    return rec["labels"] if rec else None

