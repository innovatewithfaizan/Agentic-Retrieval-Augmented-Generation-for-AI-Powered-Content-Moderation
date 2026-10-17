"""Command line entry point: python -m subrating <command>.

    match      match subtitle files to reviews -> data/matched_films.csv
    index      embed subtitle chunks into the local ChromaDB store (.cache/chroma)
    search     try a semantic or exact-word search on one film
"""

import argparse
import json

from . import config, index, reviews


def cmd_match(args) -> None:
    matches, summary = reviews.match_films()
    reviews.write_matches(matches)
    print(json.dumps(summary, indent=2))
    print(f"wrote {config.MATCHED_FILMS_CSV.relative_to(config.ROOT)}")


def cmd_index(args) -> None:
    films = reviews.read_matches(include_tv=True)
    added = index.build_index(films)
    print(f"indexed {added} new chunks ({index.persistent_store().col.count()} total)")


def cmd_search(args) -> None:
    film = next((f for f in reviews.read_matches(include_tv=True) if f["film_id"] == args.film), None)
    if film is None:
        raise SystemExit(f"unknown film_id {args.film!r}; see data/matched_films.csv")
    ctx = index.FilmContext.from_dataset(film)
    if args.words:
        print(json.dumps(index.find_words(ctx.lines, args.query.split(",")), indent=2, ensure_ascii=False))
        return
    for hit in ctx.store.search(ctx.film_id, args.query, args.k):
        print(f"[{hit['start']}] {hit['id']}\n{hit['text']}\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="subrating", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("match").set_defaults(fn=cmd_match)
    sub.add_parser("index").set_defaults(fn=cmd_index)
    p = sub.add_parser("search")
    p.add_argument("film", help="film_id from data/matched_films.csv")
    p.add_argument("query", help="search text, or comma-separated words with --words")
    p.add_argument("-k", type=int, default=config.TOP_K)
    p.add_argument("--words", action="store_true", help="exact word counts instead of semantic search")
    p.set_defaults(fn=cmd_search)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
