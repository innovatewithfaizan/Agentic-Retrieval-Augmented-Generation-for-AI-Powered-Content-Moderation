"""Write results/metrics.json, results/metrics.md and confusion-matrix figures."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

MODE_NAMES = {
    "full": "Full transcript (no retrieval)",
    "rag": "RAG, fixed query per category",
    "agentic": "Agentic RAG (search tools)",
}


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def markdown(report: dict) -> str:
    out = ["# Evaluation results", ""]
    modes = report["modes"]
    out += [
        f"All rows are scored on the same {report['films_compared']} films (labelled, and rated by every "
        f"method shown). Labels extracted from the reviews with `{report['label_model']}`.",
        "",
        "## Overall (4 categories pooled)",
        "",
        "| Method | Model | Films | Exact | 95% CI | Within one level | Weighted κ | Evidence found in subtitles | Mean tool calls | List-price cost |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mode, r in modes.items():
        o = r["overall"]
        lo, hi = o["exact_ci95"]
        out.append(
            f"| {MODE_NAMES[mode]} | `{r['model']}` | {r['films']} | **{pct(o['exact'])}** | "
            f"{pct(lo)}–{pct(hi)} | {pct(o['within_one'])} | {o['kappa_qw']:.2f} | "
            f"{pct(r['evidence_grounded'])} of {r['evidence_quotes']} | {r['mean_tool_calls']:.1f} | "
            f"${r['cost_usd']:.2f} |"
        )
    mb = report["majority_baseline"]["overall"]
    out.append(
        f"| Majority label per category | — | — | {pct(mb['exact'])} | — | "
        f"{pct(mb['within_one'])} | {mb['kappa_qw']:.2f} | — | — | — |"
    )
    out += ["", "## Exact agreement by category", ""]
    header = "| Category | Majority baseline | " + " | ".join(MODE_NAMES[m] for m in modes) + " |"
    out += [header, "|---" * (2 + len(modes)) + "|"]
    for cat, column in config.CATEGORIES.items():
        base = report["majority_baseline"]["categories"][cat]
        cells = [
            f"{pct(r['categories'][cat]['exact'])} (κ {r['categories'][cat]['kappa_qw']:.2f})"
            for r in modes.values()
        ]
        out.append(f"| {column} | {pct(base['exact'])} ({base['predicts']}) | " + " | ".join(cells) + " |")

    out += ["", "## Review label distribution", "", "| Category | none | mild | moderate | strong | from empty field |", "|---|---|---|---|---|---|"]
    for cat, column in config.CATEGORIES.items():
        d = report["label_distribution"][cat]
        c = d["counts"]
        out.append(f"| {column} | {c['none']} | {c['mild']} | {c['moderate']} | {c['strong']} | {d['from_empty_field']} |")

    la = report.get("labeler_agreement")
    if la:
        out += [
            "",
            "## Label reliability",
            "",
            f"A second, independent label pass with `{la['second_labeler']}` agreed with the main labels "
            f"on {pct(la['exact'])} of {la['n']} film-category pairs (within one level: {pct(la['within_one'])}, "
            f"weighted κ {la['kappa_qw']:.2f}).",
        ]
    out += ["", "## Mean signed error (prediction minus label, in levels)", "", "| Category | " + " | ".join(MODE_NAMES[m] for m in modes) + " |", "|---" * (1 + len(modes)) + "|"]
    for cat, column in config.CATEGORIES.items():
        out.append(f"| {column} | " + " | ".join(f"{r['categories'][cat]['mean_signed_error']:+.2f}" for r in modes.values()) + " |")
    out += ["", "Negative means the pipeline rates milder than the review.", ""]
    return "\n".join(out)


def confusion_figure(mode: str, result: dict, path) -> None:
    fig, axes = plt.subplots(1, len(config.CATEGORIES), figsize=(16, 4.2))
    for ax, (cat, column) in zip(axes, config.CATEGORIES.items()):
        m = result["categories"][cat]["confusion"]
        ax.imshow(m, cmap="Blues")
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                ax.text(j, i, v, ha="center", va="center", color="white" if v > max(map(max, m)) / 2 else "black")
        ax.set_xticks(range(4), config.SEVERITIES, rotation=30)
        ax.set_yticks(range(4), config.SEVERITIES)
        ax.set_xlabel("pipeline")
        ax.set_ylabel("review label")
        ax.set_title(f"{column}\nexact {pct(result['categories'][cat]['exact'])}", fontsize=10)
    fig.suptitle(f"{MODE_NAMES[mode]} ({result['model']}, {result['films']} films)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write(report: dict) -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (config.RESULTS_DIR / "metrics.md").write_text(markdown(report), encoding="utf-8")
    for mode, result in report["modes"].items():
        confusion_figure(mode, result, config.RESULTS_DIR / f"confusion_{mode}.png")
