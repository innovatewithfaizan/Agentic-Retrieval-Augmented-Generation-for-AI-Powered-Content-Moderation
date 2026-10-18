"""Turn free-text parent-guide reviews into severity labels.

Each review describes a film per category in a sentence or two ("Brutal
boxing matches with graphic and very serious injuries."). An LLM maps that
text to the same none/mild/moderate/strong rubric the rater uses. It sees only
the review text, never subtitles or rater output, so the labels are
independent of the pipeline being scored.
"""

import json

from . import config, llm
from .rater import rubric_text

SUBMIT = "submit_labels"

SYSTEM = f"""You convert parent-guide review text into severity labels on a fixed rubric.

<rubric>
{rubric_text()}
</rubric>

Rules:
- Use only the review text you are given. Do not use outside knowledge of the film.
- Label each category from its own field first. Use the "Parents Need to Know" summary to break ties or fill in detail.
- If a category's field is empty and the summary says nothing about that category, label it "none" and set source to "empty_field".
- Quote the few words from the review that decided the label in basis.
- Give your answer by calling {SUBMIT}."""


def label_schema() -> dict:
    per_category = {
        "type": "object",
        "properties": {
            "basis": {"type": "string"},
            "source": {"type": "string", "enum": ["category_field", "summary", "empty_field"]},
            "severity": {"type": "string", "enum": config.SEVERITIES},
        },
        "required": ["basis", "source", "severity"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {cat: per_category for cat in config.CATEGORIES},
        "required": list(config.CATEGORIES),
        "additionalProperties": False,
    }


SUBMIT_TOOL = llm.function_tool(SUBMIT, "Submit the label for every category.", label_schema())


def review_prompt(review: dict) -> str:
    fields = []
    for cat, column in config.CATEGORIES.items():
        text = review.get(column, "").strip() or "(empty)"
        fields.append(f'<field category="{cat}" column="{column}">\n{text}\n</field>')
    summary = review.get("Parents Need to Know", "").strip() or "(empty)"
    return "\n".join(fields) + f"\n<summary>\n{summary}\n</summary>\n\nLabel this review."


def validate(labels: dict) -> dict:
    for cat in config.CATEGORIES:
        if labels.get(cat, {}).get("severity") not in config.SEVERITIES:
            raise ValueError(f"{cat}: bad severity")
    return {cat: labels[cat] for cat in config.CATEGORIES}


def label_review(film_id: str, review: dict, backend: llm.Backend, attempts: int = 3) -> dict:
    usage = llm.Usage()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": review_prompt(review)}]
    error = None
    for _ in range(attempts):
        response = backend.chat(messages, [SUBMIT_TOOL], llm.force(SUBMIT))
        usage.add(response)
        calls = response.choices[0].message.tool_calls or []
        try:
            if not calls:
                raise ValueError("no tool call")
            labels = validate(llm.parse_args(calls[0]))
            break
        except (ValueError, json.JSONDecodeError) as exc:
            error = exc
    else:
        raise RuntimeError(f"labelling failed: {error}")
    return {
        "film_id": film_id,
        "provider": backend.provider,
        "model": backend.model,
        "labels": labels,
        "usage": usage.to_dict(backend.model),
    }


def read_labels(path=config.LABELS_JSONL) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {r["film_id"]: r for r in rows}
