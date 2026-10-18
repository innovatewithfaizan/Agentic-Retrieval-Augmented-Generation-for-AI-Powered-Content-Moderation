"""Prompts, schemas, the backend and the agent loop, with a fake LLM (no API calls)."""

import json
from types import SimpleNamespace

import pytest

from subrating import config, index, llm, rater

FINAL = {cat: {"rationale": "r", "evidence": [], "severity": "mild"} for cat in config.CATEGORIES}


class FakeStore:
    def __init__(self):
        self.queries = []

    def search(self, film_id, query, k, exclude=frozenset()):
        self.queries.append(query)
        return [{"id": f"{film_id}:c{len(self.queries):04d}", "start": "00:00:01", "text": "hello"}]


class FakeBackend:
    provider, model, effort = "groq", "openai/gpt-oss-120b", "medium"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []

    def chat(self, messages, tools, tool_choice):
        self.sent.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice})
        return next(self.replies)


def ctx(film_id="saw", subtitle_file="74_119118_161992_saw.xml"):
    return index.FilmContext(film_id, index.film_lines(subtitle_file), FakeStore())


def call(name, args, call_id):
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=raw))


def reply(calls=None, content=""):
    message = SimpleNamespace(content=content, tool_calls=calls)
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=100)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def submit(i="s"):
    return reply([call(rater.SUBMIT, FINAL, i)])


def _strict_ok(schema):
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        for sub in schema["properties"].values():
            _strict_ok(sub)
    if schema.get("type") == "array":
        _strict_ok(schema["items"])


def test_tool_schemas_are_strict():
    for tool in rater.SEARCH_TOOLS + [rater.SUBMIT_TOOL]:
        assert tool["function"]["strict"] is True
        _strict_ok(tool["function"]["parameters"])


def test_prompts_never_contain_the_title():
    c = ctx("million_dollar_baby", "110_97723_133067_million_dollar_baby.xml")
    for mode in ("rag", "full"):
        prompt = rater.build_prompt(c, mode)
        assert "Million Dollar Baby" not in prompt and "million_dollar_baby" not in prompt


def test_rag_prompt_runs_one_query_per_category():
    c = ctx()
    prompt = rater.build_prompt(c, "rag")
    assert c.store.queries == list(config.CATEGORY_QUERIES.values())
    assert prompt.count("<excerpts category=") == len(config.CATEGORIES)
    assert len(c.seen) == len(config.CATEGORIES)


def test_full_prompt_contains_whole_transcript():
    prompt = rater.build_prompt(ctx(), "full")
    assert prompt.count("\n") > 1000
    assert "What the fuck is going on?" in prompt


def test_single_call_mode_forces_submit_and_tracks_cost():
    backend = FakeBackend([submit()])
    rec = rater.rate(ctx(), "full", backend)
    assert rec["ratings"] == FINAL
    assert backend.sent[0]["tools"] == [rater.SUBMIT_TOOL]
    assert backend.sent[0]["tool_choice"] == llm.force(rater.SUBMIT)
    assert rec["usage"]["calls"] == 1
    assert rec["usage"]["cost_usd"] == pytest.approx((1000 * 0.15 + 100 * 0.60) / 1e6)


def test_agent_loop_runs_tools_then_submits():
    backend = FakeBackend([reply([call("find_words", {"words": ["fuck*"]}, "t1")]), submit()])
    steps = []
    rec = rater.rate(ctx(), "agentic", backend, on_step=steps.append)

    assert rec["ratings"] == FINAL
    assert rec["tool_calls"][0]["tool"] == "find_words"
    assert rec["tool_calls"][0]["counts"]["fuck*"] >= 28
    assert steps == rec["tool_calls"]
    assert backend.sent[0]["tool_choice"] == "auto"
    assert len(backend.sent[0]["tools"]) == 3
    # the second request carries the assistant tool call and its result
    msgs = backend.sent[1]["messages"]
    assert msgs[-2]["role"] == "assistant" and msgs[-2]["tool_calls"][0]["id"] == "t1"
    assert msgs[-1] == {"role": "tool", "tool_call_id": "t1", "content": msgs[-1]["content"]}
    assert json.loads(msgs[-1]["content"])["counts"]["fuck*"] >= 28


def test_agent_search_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(config, "AGENT_MAX_SEARCHES", 2)
    calls = [call("find_words", {"words": ["gun"]}, f"t{i}") for i in range(3)]
    backend = FakeBackend([reply(calls), submit()])
    rec = rater.rate(ctx(), "agentic", backend)

    tool_msgs = [m for m in backend.sent[1]["messages"] if m["role"] == "tool"]
    assert "budget used up" in tool_msgs[2]["content"]
    assert backend.sent[1]["tool_choice"] == llm.force(rater.SUBMIT)
    assert [t.get("error") for t in rec["tool_calls"]] == [None, None, "budget"]


def test_invalid_submission_is_sent_back_for_a_retry():
    bad = {**FINAL, "violence": {"rationale": "r", "evidence": [], "severity": "extreme"}}
    backend = FakeBackend([reply([call(rater.SUBMIT, bad, "s1")]), submit("s2")])
    rec = rater.rate(ctx(), "full", backend)
    assert rec["ratings"] == FINAL
    assert "Invalid ratings" in backend.sent[1]["messages"][-1]["content"]


def test_plain_text_json_answer_is_accepted():
    backend = FakeBackend([reply(None, content="Here you go: " + json.dumps(FINAL))])
    assert rater.rate(ctx(), "agentic", backend)["ratings"] == FINAL


def test_backend_reasoning_settings_per_provider(monkeypatch):
    groq = llm.Backend("groq", "openai/gpt-oss-120b", api_key="test-key", effort="low")
    assert groq.extra_body() == {"reasoning_effort": "low"}
    assert llm.Backend("groq", "qwen/qwen3.8-27b", api_key="test-key").extra_body() == {"reasoning_format": "hidden"}
    router = llm.Backend("openrouter", "openai/gpt-oss-120b", api_key="test-key", effort="high")
    assert router.extra_body() == {"reasoning": {"effort": "high"}}
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TOGETHER_API_KEY"):
        llm.Backend("together", "openai/gpt-oss-120b")


def test_usage_cost_uses_model_prices():
    u = llm.Usage(calls=1, input_tokens=1_000_000, output_tokens=100_000)
    assert u.cost("openai/gpt-oss-120b") == pytest.approx(0.15 + 0.06)
    assert u.cost("unknown/model") == 0.0
