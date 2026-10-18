"""Compare pipeline ratings with review-derived labels.

Metrics per category and pooled over categories:
* exact agreement - same severity level
* within-one      - off by at most one level
* quadratic-weighted Cohen's kappa - agreement corrected for chance, with
  larger penalties for bigger misses on the ordinal scale
* a 95% bootstrap interval over films for pooled exact agreement
* evidence grounding - share of quoted evidence lines that really occur in
  the film's subtitles (a hallucination check)
"""

import json
import random
import re

import numpy as np

from . import config, index, labels
from .reviews import read_matches

LEVEL = {s: i for i, s in enumerate(config.SEVERITIES)}
K = len(config.SEVERITIES)


def confusion(pairs: list[tuple[str, str]]) -> np.ndarray:
    """Rows are review labels, columns are predictions."""
    m = np.zeros((K, K), dtype=int)
    for truth, pred in pairs:
        m[LEVEL[truth], LEVEL[pred]] += 1
    return m


def weighted_kappa(m: np.ndarray) -> float:
    n = m.sum()
    if n == 0:
        return float("nan")
    i, j = np.indices((K, K))
    w = (i - j) ** 2 / (K - 1) ** 2
    expected = np.outer(m.sum(1), m.sum(0)) / n
    denom = (w * expected).sum()
    return float(1 - (w * m).sum() / denom) if denom else float("nan")


def summarize(pairs: list[tuple[str, str]]) -> dict:
    m = confusion(pairs)
    n = int(m.sum())
    diffs = [abs(LEVEL[t] - LEVEL[p]) for t, p in pairs]
    return {
        "n": n,
        "exact": sum(d == 0 for d in diffs) / n if n else float("nan"),
        "within_one": sum(d <= 1 for d in diffs) / n if n else float("nan"),
        "kappa_qw": weighted_kappa(m),
        "mean_signed_error": float(np.mean([LEVEL[p] - LEVEL[t] for t, p in pairs])) if n else 0.0,
        "confusion": m.tolist(),
    }


def bootstrap_exact(per_film: dict[str, list[bool]], iters: int = 2000, seed: int = 0):
    films = list(per_film)
    rng = random.Random(seed)
    scores = []
    for _ in range(iters):
        sample = [rng.choice(films) for _ in films]
        hits = [h for f in sample for h in per_film[f]]
        scores.append(sum(hits) / len(hits))
    scores.sort()
    return scores[int(0.025 * iters)], scores[int(0.975 * iters) - 1]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def grounding(record: dict, subtitle_file: str) -> tuple[int, int]:
    transcript = _norm(" ".join(line.text for line in index.film_lines(subtitle_file)))
    found = total = 0
    for cat in config.CATEGORIES:
        for quote in record["ratings"][cat]["evidence"]:
            # Strip a leading "[00:12:34]" if the model copied the timestamp.
            q = _norm(re.sub(r"^\[\d\d:\d\d:\d\d\]\s*", "", quote))
            if not q:
                continue
            total += 1
            found += q in transcript
    return found, total


def load_ratings(model: str, mode: str) -> dict[str, dict]:
    out = {}
    for path in sorted(config.ratings_dir(model, mode).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        out[rec["film_id"]] = rec
    return out


def evaluate_mode(model: str, mode: str, films: list[dict], gold: dict[str, dict]) -> dict | None:
    ratings = load_ratings(model, mode)
    films = [f for f in films if f["film_id"] in ratings and f["film_id"] in gold]
    if not films:
        return None
    by_cat = {cat: [] for cat in config.CATEGORIES}
    per_film = {}
    found = total = 0
    cost = 0.0
    tool_calls = []
    for f in films:
        rec, lab = ratings[f["film_id"]], gold[f["film_id"]]["labels"]
        per_film[f["film_id"]] = []
        for cat in config.CATEGORIES:
            pair = (lab[cat]["severity"], rec["ratings"][cat]["severity"])
            by_cat[cat].append(pair)
            per_film[f["film_id"]].append(pair[0] == pair[1])
        g_found, g_total = grounding(rec, f["subtitle_file"])
        found, total = found + g_found, total + g_total
        cost += rec["usage"]["cost_usd"]
        tool_calls.append(len(rec.get("tool_calls", [])))
    pooled = [p for pairs in by_cat.values() for p in pairs]
    lo, hi = bootstrap_exact(per_film)
    return {
        "mode": mode,
        "model": next(iter(ratings.values()))["model"],
        "films": len(films),
        "overall": {**summarize(pooled), "exact_ci95": [lo, hi]},
        "categories": {cat: summarize(pairs) for cat, pairs in by_cat.items()},
        "evidence_grounded": found / total if total else float("nan"),
        "evidence_quotes": total,
        "cost_usd": round(cost, 4),
        "mean_tool_calls": float(np.mean(tool_calls)),
    }


def majority_baseline(films: list[dict], gold: dict[str, dict]) -> dict:
    """Predict each category's most common review label for every film.

    This peeks at the label distribution, so it is an optimistic floor."""
    by_cat = {}
    pooled = []
    for cat in config.CATEGORIES:
        truth = [gold[f["film_id"]]["labels"][cat]["severity"] for f in films if f["film_id"] in gold]
        top = max(config.SEVERITIES, key=truth.count)
        pairs = [(t, top) for t in truth]
        by_cat[cat] = {**summarize(pairs), "predicts": top}
        pooled += pairs
    return {"overall": summarize(pooled), "categories": by_cat}


def label_distribution(films, gold) -> dict:
    dist = {}
    for cat in config.CATEGORIES:
        sev = [gold[f["film_id"]]["labels"][cat]["severity"] for f in films if f["film_id"] in gold]
        src = [gold[f["film_id"]]["labels"][cat]["source"] for f in films if f["film_id"] in gold]
        dist[cat] = {
            "counts": {s: sev.count(s) for s in config.SEVERITIES},
            "from_empty_field": src.count("empty_field"),
        }
    return dist


def labeler_agreement(films, gold, check_path) -> dict | None:
    other = labels.read_labels(check_path)
    if not other:
        return None
    pairs = [
        (gold[f["film_id"]]["labels"][cat]["severity"], other[f["film_id"]]["labels"][cat]["severity"])
        for f in films
        if f["film_id"] in gold and f["film_id"] in other
        for cat in config.CATEGORIES
    ]
    model = next(iter(other.values()))["model"]
    return {"second_labeler": model, **summarize(pairs)}


def run(model: str = config.RATER_MODEL, modes=("full", "rag", "agentic")) -> dict:
    """Score every mode on the same films: those with labels and with ratings
    from every mode that has any ratings at all."""
    gold = labels.read_labels()
    films = [f for f in read_matches() if f["film_id"] in gold]
    present = {m: load_ratings(model, m) for m in modes}
    present = {m: r for m, r in present.items() if r}
    common = [f for f in films if all(f["film_id"] in r for r in present.values())]
    report = {
        "films_with_labels": len(films),
        "films_compared": len(common),
        "ratings_available": {m: len(r) for m, r in present.items()},
        "label_model": next(iter(gold.values()))["model"] if gold else None,
        "label_distribution": label_distribution(common, gold),
        "majority_baseline": majority_baseline(common, gold),
        "labeler_agreement": labeler_agreement(common, gold, config.LABELS_CHECK_JSONL),
        "modes": {},
    }
    for mode in present:
        res = evaluate_mode(model, mode, common, gold)
        if res:
            report["modes"][mode] = res
    return report
