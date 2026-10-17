"""Subtitle parsing, review loading and title matching (no API, no embeddings)."""

from subrating import reviews, subtitles

XML = """<?xml version="1.0" encoding="utf-8"?>
<document>
  <s id="1"><time id="T1S" value="00:00:00,040" /><w id="1.1">23</w><w id="1.2">.</w><w id="1.3">976</w></s>
  <s id="2">
    <time id="T2S" value="00:00:49,120" />
    <w id="2.1">I'</w><w id="2.2">ve</w><w id="2.3">got</w><w id="2.4">it</w><w id="2.5">.</w>
    <time id="T2E" value="00:00:53,880" />
  </s>
  <s id="3"><time id="T3S" value="00:01:10,000" /><w id="3.1">I'</w><w id="3.2">il</w><w id="3.3">go</w><w id="3.4">,</w><w id="3.5">ok</w></s>
</document>"""


def test_clean_rejoins_contractions_and_punctuation():
    assert subtitles.clean("I' ve only wouldn' t go , ok .") == "I've only wouldn't go, ok."
    assert subtitles.clean("I' il be there") == "I'll be there"


def test_parse_xml_skips_frame_counter_and_keeps_timestamps(tmp_path):
    path = tmp_path / "film.xml"
    path.write_text(XML, encoding="utf-8")
    lines = subtitles.parse_xml(path)
    assert [line.text for line in lines] == ["I've got it.", "I'll go, ok"]
    assert [line.start for line in lines] == ["00:00:49", "00:01:10"]


def test_chunk_lines_groups_consecutive_lines():
    lines = [subtitles.Line(str(i), f"00:00:{i:02d}", f"line {i}") for i in range(10)]
    chunks = subtitles.chunk_lines(lines, size=4)
    assert [c.chunk_id for c in chunks] == ["c0000", "c0001", "c0002"]
    assert chunks[1].start == "00:00:04"
    assert chunks[2].text == "line 8\nline 9"


def test_slugs_normalise_both_sides():
    assert reviews.slugify("Ocean's Twelve  ") == "oceans_twelve"
    assert reviews.slugify("Charlie and the Chocolate Factory (2005)  ") == "charlie_and_the_chocolate_factory"
    assert reviews.slugify("Prison Break TV Review ") == "prison_break"
    assert reviews.subtitle_slug("3270_110542_150715_kate__leopold.xml") == "kate_and_leopold"


def test_reviews_parse_multiline_fields():
    rows = reviews.load_reviews()
    assert len(rows) == 83
    saw = next(r for r in rows if r["Movie"] == "Saw")
    assert "f--k" in saw["Language"]
    assert saw["Parents Need to Know"].startswith("Parents need to know")


def test_match_films_counts():
    matches, summary = reviews.match_films()
    assert summary["matched"] == 77
    assert summary["matched_films"] == 75
    assert summary["matched_tv"] == 2
    assert summary["unmatched_subtitles"] == ["2454_71730_93455_spartan.xml"]
    by_id = {m["film_id"]: m for m in matches}
    assert by_id["terminator_3_rise_of_the_machines"]["review_title"] == "Terminator 3"
    assert by_id["avp_alien_vs_predator"]["review_title"] == "Alien vs. Predator"
    assert by_id["maria_full_of_grace"]["review_title"] == "Maria"
    # every matched review is used once
    assert len({m["review_title"] for m in matches}) == 77
