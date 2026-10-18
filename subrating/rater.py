"""Rate a film's content per category from its subtitles.

Three modes, so the README can compare them on the same films:

* ``rag``     - one fixed query per category, top-k chunks, one LLM call.
* ``agentic`` - the same starting excerpts plus two search tools. The model
                decides what else to look up, within a fixed tool budget,
                before it submits ratings.
* ``full``    - no retrieval; the whole transcript in one call (long-context
                baseline).

Ratings always come back through a ``submit_ratings`` tool call with a strict
schema, which works the same way on every OpenAI-compatible provider. Film
titles are never sent to the model, so ratings come from the dialogue and not
from what the model remembers about a famous film.
"""

import json
import re
import time

from . import config, llm
from .index import FilmContext, find_words
from .subtitles import full_transcript

MODES = ("rag", "agentic", "full")
SUBMIT = "submit_ratings"
MAX_TURNS = config.AGENT_MAX_SEARCHES + 4


def rubric_text() -> str:
    parts = []
    for cat, column in config.CATEGORIES.items():
        levels = "\n".join(f"  - {sev}: {desc}" for sev, desc in config.RUBRIC[cat].items())
        parts.append(f"{cat} ({column}):\n{levels}")
    return "\n\n".join(parts)


SYSTEM = f"""You estimate how a parent guide would rate a film for parents of a 10-year-old, using only the film's subtitle dialogue.

Rate four categories on a four-level scale: none, mild, moderate, strong.

<rubric>
{rubric_text()}
</rubric>

How to rate:
- Dialogue is your only evidence, and much of what parent guides rate is visual. Infer what is likely on screen from dialogue cues (threats, screams, talk of bodies, weapons, sex, drinking), but do not invent content the dialogue does not support.
- Rate the most intense content you find. A single clear instance can set the level (one f-word makes language "strong").
- For evidence, copy up to 3 subtitle lines verbatim from the material you were given. Leave evidence empty when the severity is "none".
- The film's title is withheld on purpose. Do not try to identify the film or use outside knowledge of it.
- Give your answer by calling {SUBMIT}."""

AGENT_ADDENDUM = f"""

You also have two tools for gathering more dialogue before you answer:
- search_subtitles finds passages by meaning. Phrase queries as lines a character might say ("put the gun down", "you're drunk again").
- find_words counts exact words across the whole film. Use it for profanity, slurs, drug or weapon names.

The excerpts you start with come from one generic query per category and can miss things. For each category where the evidence is thin, ambiguous or close to the boundary between two levels, look further before you rate. You can make at most {config.AGENT_MAX_SEARCHES} search calls in total. Then call {SUBMIT}."""


def rating_schema() -> dict:
    per_category = {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "severity": {"type": "string", "enum": config.SEVERITIES},
        },
        "required": ["rationale", "evidence", "severity"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {cat: per_category for cat in config.CATEGORIES},
        "required": list(config.CATEGORIES),
        "additionalProperties": False,
    }


SUBMIT_TOOL = llm.function_tool(SUBMIT, "Submit the final rating for every category.", rating_schema())
SEARCH_TOOLS = [
    llm.function_tool(
        "search_subtitles",
        "Semantic search over this film's subtitle dialogue (all-MiniLM-L6-v2 embeddings of "
        f"{config.CHUNK_SIZE}-line chunks). Returns up to {config.AGENT_SEARCH_K} chunks you have "
        "not seen yet, each with its start time.",
        {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": list(config.CATEGORIES)},
                "query": {"type": "string"},
            },
            "required": ["category", "query"],
            "additionalProperties": False,
        },
    ),
    llm.function_tool(
        "find_words",
        "Case-insensitive whole-word counts across every subtitle line of the film, with up to 8 "
        "example lines. Each word also matches common endings ('fuck' finds 'fucking', 'drunk' "
        "finds 'drunken'); a trailing * matches any ending. At most 12 words per call.",
        {
            "type": "object",
            "properties": {"words": {"type": "array", "items": {"type": "string"}}},
            "required": ["words"],
            "additionalProperties": False,
        },
    ),
]


def validate(ratings: dict) -> dict:
    out = {}
    for cat in config.CATEGORIES:
        r = ratings.get(cat)
        if not isinstance(r, dict) or r.get("severity") not in config.SEVERITIES:
            raise ValueError(f"{cat}: severity must be one of {config.SEVERITIES}")
        evidence = r.get("evidence") or []
        if not isinstance(evidence, list):
            raise ValueError(f"{cat}: evidence must be a list of strings")
        out[cat] = {
            "rationale": str(r.get("rationale", "")),
            "evidence": [str(e) for e in evidence],
            "severity": r["severity"],
        }
    return out


def _json_in_text(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON object in reply")
    return json.loads(match.group(0))


def _format_chunks(chunks: list[dict]) -> str:
    return "\n---\n".join(f"[{c['start']}]\n{c['text']}" for c in chunks)


def build_prompt(ctx: FilmContext, mode: str) -> str:
    if mode == "full":
        return f"<transcript>\n{full_transcript(ctx.lines)}\n</transcript>\n\nRate this film."
    blocks = []
    for cat, query in config.CATEGORY_QUERIES.items():
        chunks = ctx.store.search(ctx.film_id, query, config.TOP_K)
        ctx.seen.update(c["id"] for c in chunks)
        blocks.append(f'<excerpts category="{cat}">\n{_format_chunks(chunks)}\n</excerpts>')
    intro = (
        "Below are subtitle excerpts retrieved for each category. Each excerpt is "
        f"{config.CHUNK_SIZE} consecutive lines with its start time."
    )
    return f"{intro}\n\n" + "\n\n".join(blocks) + "\n\nRate this film."


def run_tool(ctx: FilmContext, name: str, args: dict) -> tuple[str, dict]:
    if name == "search_subtitles":
        hits = ctx.store.search(ctx.film_id, args.get("query", ""), config.AGENT_SEARCH_K, ctx.seen)
        ctx.seen.update(h["id"] for h in hits)
        return (_format_chunks(hits) if hits else "(no new passages)"), {"new_chunks": len(hits)}
    if name == "find_words":
        found = find_words(ctx.lines, list(args.get("words", [])))
        return json.dumps(found, ensure_ascii=False), {"counts": found["counts"]}
    return f"Unknown tool {name}.", {"error": "unknown tool"}


def rate(ctx: FilmContext, mode: str, backend: llm.Backend, on_step=None) -> dict:
    """Rate one film. on_step(event: dict) is called for each tool call (used by the app)."""
    t0 = time.time()
    usage = llm.Usage()
    agentic = mode == "agentic"
    messages = [
        {"role": "system", "content": SYSTEM + (AGENT_ADDENDUM if agentic else "")},
        {"role": "user", "content": build_prompt(ctx, mode)},
    ]
    trace, searches, ratings = [], 0, None

    for _ in range(MAX_TURNS):
        can_search = agentic and searches < config.AGENT_MAX_SEARCHES
        tools = SEARCH_TOOLS + [SUBMIT_TOOL] if agentic else [SUBMIT_TOOL]
        response = backend.chat(messages, tools, "auto" if can_search else llm.force(SUBMIT))
        usage.add(response)
        message = response.choices[0].message
        calls = message.tool_calls or []

        if not calls:  # some models answer in text despite the tools; accept valid JSON
            try:
                ratings = validate(_json_in_text(message.content))
                break
            except (ValueError, json.JSONDecodeError):
                messages.append(llm.assistant_message(message))
                messages.append({"role": "user", "content": f"Call {SUBMIT} with your ratings."})
                continue

        messages.append(llm.assistant_message(message))
        for call in calls:
            name = call.function.name
            try:
                args = llm.parse_args(call)
            except json.JSONDecodeError as exc:
                args, content, info = {}, f"Arguments were not valid JSON: {exc}", {"error": "bad json"}
            else:
                if name == SUBMIT:
                    try:
                        ratings = validate(args)
                        content, info = "Received.", {}
                    except ValueError as exc:
                        content, info = f"Invalid ratings: {exc}. Call {SUBMIT} again.", {"error": str(exc)}
                elif searches >= config.AGENT_MAX_SEARCHES or not agentic:
                    content, info = f"Search budget used up. Call {SUBMIT} now.", {"error": "budget"}
                else:
                    content, info = run_tool(ctx, name, args)
                    searches += 1
            if name != SUBMIT:
                event = {"tool": name, "input": args, **info}
                trace.append(event)
                if on_step:
                    on_step(event)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        if ratings is not None:
            break
    if ratings is None:
        raise RuntimeError(f"no valid ratings after {MAX_TURNS} turns")

    return {
        "film_id": ctx.film_id,
        "mode": mode,
        "provider": backend.provider,
        "model": backend.model,
        "effort": backend.effort,
        "ratings": ratings,
        "tool_calls": trace,
        "usage": usage.to_dict(backend.model),
        "seconds": round(time.time() - t0, 1),
    }
