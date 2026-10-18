"""Command line entry point: python -m subrating <command>.

    match      match subtitle files to reviews -> data/matched_films.csv
    index      embed subtitle chunks into the local ChromaDB store (.cache/chroma)
    search     try a semantic or exact-word search on one film
    estimate   token and cost estimate for the rating modes (no API calls)
    rate       rate films from their subtitles with an LLM (needs an API key)
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, index, llm, rater, reviews


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


def select_films(args) -> list[dict]:
    films = reviews.read_matches(include_tv=args.include_tv)
    if args.films:
        wanted = set(args.films.split(","))
        films = [f for f in films if f["film_id"] in wanted]
    return films[: args.limit] if args.limit else films


def cmd_estimate(args) -> None:
    """Rough token counts (chars / 4) and list-price cost for each rating mode.

    Output tokens and the agent's extra turns are assumptions; replace them
    with the measured numbers from a pilot run before a full run."""
    films = select_films(args)
    out_per_answer, agent_turns, tool_result_tokens = 2000, 3, 2500
    totals = {mode: [0, 0] for mode in rater.MODES}
    for film in films:
        ctx = index.FilmContext.from_dataset(film)
        for mode in ("rag", "full"):
            totals[mode][0] += len(rater.SYSTEM + rater.build_prompt(ctx, mode)) // 4
            totals[mode][1] += out_per_answer
        base = len(rater.SYSTEM + rater.AGENT_ADDENDUM + rater.build_prompt(ctx, "rag")) // 4
        totals["agentic"][0] += agent_turns * base + (agent_turns - 1) * tool_result_tokens
        totals["agentic"][1] += agent_turns * out_per_answer
    price_in, price_out = llm.PRICES.get(args.model, (0.0, 0.0))
    print(f"{len(films)} films, {args.model} at ${price_in}/${price_out} per M tokens (approximate)")
    grand = 0.0
    for mode, (inp, out) in totals.items():
        cost = (inp * price_in + out * price_out) / 1e6
        grand += cost
        print(f"  {mode:<8} {inp:>10,} in {out:>9,} out  ${cost:.2f}")
    print(f"  {'total':<8} ${grand:.2f}")


def cmd_rate(args) -> None:
    try:
        backend = llm.Backend(args.provider, args.model, effort=args.effort)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    out_dir = config.RATINGS_DIR / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    films = select_films(args)
    todo = [f for f in films if not (out_dir / f"{f['film_id']}.json").exists()]
    print(f"rating {len(todo)} films: mode={args.mode} {args.provider}/{args.model} "
          f"({len(films) - len(todo)} already done)")

    def work(film):
        rec = rater.rate(index.FilmContext.from_dataset(film), args.mode, backend)
        path = out_dir / f"{film['film_id']}.json"
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        return rec

    spent, failed = 0.0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, f): f["film_id"] for f in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            fid = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # keep going, report at the end
                failed.append(fid)
                print(f"[{n}/{len(todo)}] {fid}: ERROR {exc!r}", flush=True)
                continue
            spent += rec["usage"]["cost_usd"]
            sev = " ".join(f"{c}={r['severity']}" for c, r in rec["ratings"].items())
            print(f"[{n}/{len(todo)}] {fid}: {sev} tools={len(rec['tool_calls'])} "
                  f"${rec['usage']['cost_usd']:.4f}", flush=True)
    print(f"spent ${spent:.4f}" + (f"; failed: {', '.join(failed)}" if failed else ""))


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

    def film_args(p):
        p.add_argument("--provider", default=config.PROVIDER, choices=list(llm.PROVIDERS))
        p.add_argument("--model", default=config.RATER_MODEL)
        p.add_argument("--films", help="comma-separated film_ids (default: all 75 films)")
        p.add_argument("--limit", type=int, help="only the first N films, for a pilot run")
        p.add_argument("--include-tv", action="store_true", help="also rate the 2 TV-series episodes")

    p = sub.add_parser("estimate")
    film_args(p)
    p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser("rate")
    film_args(p)
    p.add_argument("--mode", choices=rater.MODES, required=True)
    p.add_argument("--effort", default=config.EFFORT, choices=["low", "medium", "high"])
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(fn=cmd_rate)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
