"""Label extraction with a fake backend, and the CLI's daily-limit detection."""

import json
from types import SimpleNamespace

from subrating import config, labels, reviews
from subrating.__main__ import daily_limit_hit

LABELS = {cat: {"basis": "b", "source": "category_field", "severity": "strong"} for cat in config.CATEGORIES}


class FakeBackend:
    provider, model, effort = "groq", "qwen/qwen3.8-27b", "medium"

    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.sent = []

    def chat(self, messages, tools, tool_choice):
        self.sent.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        call = SimpleNamespace(id="c", function=SimpleNamespace(name=labels.SUBMIT, arguments=json.dumps(next(self.payloads))))
        message = SimpleNamespace(content="", tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=500, completion_tokens=50))


def saw_review():
    return reviews.reviews_by_title()["Saw"]


def test_review_prompt_has_every_category_and_no_subtitles():
    prompt = labels.review_prompt(saw_review())
    for column in config.CATEGORIES.values():
        assert f'column="{column}"' in prompt
    assert "f--k" in prompt and "<summary>" in prompt


def test_label_review_retries_until_valid():
    bad = {**LABELS, "sex": {"basis": "b", "source": "summary", "severity": "R-rated"}}
    backend = FakeBackend([bad, LABELS])
    rec = labels.label_review("saw", saw_review(), backend)
    assert rec["labels"] == LABELS
    assert rec["usage"]["calls"] == 2
    assert backend.sent[0]["tool_choice"]["function"]["name"] == labels.SUBMIT


def test_daily_limit_detection():
    assert daily_limit_hit(Exception("Rate limit reached ... on tokens per day (TPD): Limit 200000"))
    assert daily_limit_hit(Exception("... on requests per day (RPD): Limit 1000"))
    assert not daily_limit_hit(Exception("... on tokens per minute (TPM): Limit 8000"))
