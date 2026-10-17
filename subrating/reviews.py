"""Load the parent-guide review CSVs and match them to subtitle files.

Subtitle files are named ``<id>_<id>_<id>_<title_slug>.xml``. Review titles
come from a scrape and are messy: trailing spaces, "(2005)", " TV Review",
and titles cut at a colon ("Terminator 3" for "Terminator 3: Rise of the
Machines"). Both sides are normalised to slugs and matched exactly, then by a
whole-word prefix or suffix when the exact match fails.
"""

import csv
import re
from pathlib import Path

from . import config


def slugify(title: str) -> str:
    s = title.lower().strip()
    s = re.sub(r"\btv review\b", "", s)
    s = re.sub(r"\(\d{4}\)", "", s)
    s = s.replace("&", " and ").replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def subtitle_slug(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^\d+_\d+_\d+_", "", stem)
    return stem.replace("__", "_and_")  # "kate__leopold" lost its "&"


def load_reviews() -> list[dict]:
    rows = []
    for path in sorted(config.REVIEW_DIR.glob("*.csv")):
        # Fields contain embedded newlines, so use the csv module, not split().
        with open(path, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                row = {k: (v or "").strip() for k, v in row.items()}
                row["genre"] = path.stem
                rows.append(row)
    return rows


def match_films() -> tuple[list[dict], dict]:
    reviews = load_reviews()
    by_slug = {}
    for r in reviews:
        if r["Movie"] and r["Movie"] != "view-source":  # one scraped junk row
            by_slug[slugify(r["Movie"])] = r

    matches, unmatched_subs = [], []
    for sub in sorted(p.name for p in config.SUBTITLE_DIR.glob("*.xml")):
        slug = subtitle_slug(sub)
        review, how = by_slug.get(slug), "exact"
        if review is None:
            prefix = [k for k in by_slug if slug.startswith(k + "_")]
            suffix = [k for k in by_slug if slug.endswith("_" + k)]
            if len(prefix) == 1:
                review, how = by_slug[prefix[0]], "prefix"
            elif len(suffix) == 1:
                review, how = by_slug[suffix[0]], "suffix"
        if review is None:
            unmatched_subs.append(sub)
            continue
        matches.append(
            {
                "film_id": slug,
                "subtitle_file": sub,
                "review_title": review["Movie"],
                "genre": review["genre"],
                "kind": "tv" if "TV Review" in review["Movie"] else "film",
                "match": how,
            }
        )

    matched_titles = {m["review_title"] for m in matches}
    summary = {
        "subtitle_files": len(list(config.SUBTITLE_DIR.glob("*.xml"))),
        "review_rows": len(reviews),
        "matched": len(matches),
        "matched_films": sum(m["kind"] == "film" for m in matches),
        "matched_tv": sum(m["kind"] == "tv" for m in matches),
        "unmatched_subtitles": unmatched_subs,
        "unmatched_reviews": sorted(
            r["Movie"] for r in reviews if r["Movie"] not in matched_titles
        ),
    }
    return matches, summary


def write_matches(matches: list[dict]) -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    with open(config.MATCHED_FILMS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(matches[0]))
        writer.writeheader()
        writer.writerows(matches)


def read_matches(include_tv: bool = False) -> list[dict]:
    with open(config.MATCHED_FILMS_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if include_tv or r["kind"] == "film"]


def reviews_by_title() -> dict[str, dict]:
    return {r["Movie"]: r for r in load_reviews()}
