"""The demo app's helpers: key handling, uploads and the no-key view."""

import pytest

from subrating import config, demo, index, subtitles

SRT = (
    "﻿1\r\n00:00:01,000 --> 00:00:03,000\r\n<i>Put the gun down.</i>\r\n\r\n"
    "2\r\n00:01:02,500 --> 00:01:04,000\r\n{\\an8}I' m not\r\ngoing anywhere .\r\n\r\n"
    "3\r\n00:01:05,000 --> 00:01:06,000\r\n\r\n"
)


def test_parse_srt_strips_tags_and_keeps_start_times():
    lines = subtitles.parse_srt(SRT)
    assert [line.text for line in lines] == ["Put the gun down.", "I'm not going anywhere."]
    assert [line.start for line in lines] == ["00:00:01", "00:01:02"]


def test_parse_upload_accepts_srt_and_xml_and_rejects_others():
    assert len(demo.parse_upload("film.SRT", SRT.encode("utf-8"))) == 2
    xml = (config.SUBTITLE_DIR / "110_97723_133067_million_dollar_baby.xml").read_bytes()
    assert len(demo.parse_upload("film.xml", xml)) > 500
    with pytest.raises(ValueError):
        demo.parse_upload("film.txt", b"hello")
    with pytest.raises(ValueError):
        demo.parse_upload("film.srt", b"no timestamps here")


def test_key_comes_from_the_visitor_unless_dev_mode(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "env-key")
    assert demo.resolve_key("groq", "  typed  ", dev=False) == "typed"
    assert demo.resolve_key("groq", "", dev=False) == ""  # never the owner's key
    assert demo.resolve_key("groq", "", dev=True) == "env-key"
    assert demo.resolve_key("groq", "typed", dev=True) == "typed"


def test_backend_needs_an_explicit_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    with pytest.raises(ValueError):
        demo.make_backend("openai", "gpt-5-mini", "", "medium")
    assert demo.make_backend("openai", "gpt-5-mini", "visitor-key", "medium").api_key == "visitor-key"


def test_uploads_get_a_private_id_and_are_forgotten():
    store = index.memory_store()
    ctx = demo.upload_context(subtitles.parse_srt(SRT), store)
    assert ctx.film_id.startswith("upload-")
    assert store.search(ctx.film_id, "gun", k=1)
    demo.forget(ctx)
    assert not store.col.get(where={"film_id": ctx.film_id})["ids"]


def test_saved_ratings_and_labels_are_available_without_a_key():
    rec = demo.precomputed("million_dollar_baby", "agentic")
    assert rec["model"] == demo.DEMO_MODEL and set(rec["ratings"]) == set(config.CATEGORIES)
    assert set(demo.review_labels("million_dollar_baby")) == set(config.CATEGORIES)
    assert demo.precomputed("no_such_film", "rag") is None


def test_app_runs_without_a_key(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(demo, "DEV_ENV_KEYS", False)
    at = AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.button[0].disabled  # nothing can be sent without a key
    assert any("Parent-guide label" in c.value for c in at.caption)
