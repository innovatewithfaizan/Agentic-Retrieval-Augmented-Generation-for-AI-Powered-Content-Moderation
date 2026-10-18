"""Exact word search used by the agent (no embeddings needed)."""

from subrating import index
from subrating.subtitles import Line


def test_find_words_counts_whole_words_with_wildcard():
    found = index.find_words(index.film_lines("74_119118_161992_saw.xml"), ["fuck*", "shit", "zzzz"])
    assert found["counts"]["fuck*"] >= 28  # fuck, fucking, fuckin
    assert found["counts"]["shit"] >= 10
    assert found["counts"]["zzzz"] == 0
    assert 0 < len(found["example_lines"]) <= 8


def test_find_words_matches_common_endings_but_not_other_words():
    lines = [Line("1", "00:00:01", "What the fucking hell, you fucker."),
             Line("2", "00:00:02", "Hello there, he got drunken again."),
             Line("3", "00:00:03", "Shell game.")]
    counts = index.find_words(lines, ["fuck", "hell", "drunk"])["counts"]
    assert counts == {"fuck": 2, "hell": 1, "drunk": 1}


def test_memory_store_finds_the_relevant_chunk_and_skips_seen_ones():
    lines = [Line(str(i), f"00:00:{i:02d}", t) for i, t in enumerate([
        "Good morning, how are you?", "Fine, thanks.", "Want some coffee?", "Sure.",
        "Drop the gun or I shoot!", "He's bleeding, call an ambulance!", "Get down!", "Run!",
        "Let's go to the beach.", "I love the sea.", "Bring a towel.", "Okay.",
    ])]
    store = index.memory_store()
    assert store.ensure("demo", lines) == 3
    assert store.ensure("demo", lines) == 0  # idempotent
    top = store.search("demo", "shooting and violence", k=1)
    assert top[0]["id"] == "demo:c0001"
    rest = store.search("demo", "shooting and violence", k=2, exclude={"demo:c0001"})
    assert "demo:c0001" not in {h["id"] for h in rest} and len(rest) == 2


def test_persistent_store_is_created_once_across_threads(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(index.config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(index, "_persistent", None)
    with ThreadPoolExecutor(max_workers=8) as pool:
        stores = list(pool.map(lambda _: index.persistent_store(), range(8)))
    assert all(s is stores[0] for s in stores)
