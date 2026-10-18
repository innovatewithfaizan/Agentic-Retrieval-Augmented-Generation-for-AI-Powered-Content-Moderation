"""Streamlit demo: rate a film's content from its subtitles with your own API key.

    streamlit run app.py

The key you paste is used for one run and kept nowhere else (see
subrating/demo.py). Without a key you can still browse the committed
gpt-5-mini ratings next to the parent-guide labels.
"""

import re

import openai
import streamlit as st

from subrating import config, demo, index, llm, rater

st.set_page_config(page_title="Subtitle content rater", page_icon="🎬", layout="wide")

MARK = {"none": "⚪", "mild": "🟢", "moderate": "🟠", "strong": "🔴"}
MODE_HELP = {
    "agentic": "Agentic RAG: starts from fixed excerpts, then the model searches the subtitles itself.",
    "rag": "RAG: one fixed query per category, top 12 chunks each, one call.",
    "full": "Full transcript: the whole film in one call (about 10k tokens; too big for Groq's free tier).",
}
REPO = "https://github.com/innovatewithfaizan/Agentic-Retrieval-Augmented-Generation-for-AI-Powered-Content-Moderation"


@st.cache_resource
def store() -> index.Store:
    """One in-memory vector store per server process. Holds dataset films
    (public) and, briefly, uploads under random ids that are deleted after use."""
    return index.memory_store()


@st.cache_data
def film_list() -> list[dict]:
    return demo.films()


def show_rating(rec: dict, gold: dict | None) -> None:
    cols = st.columns(len(config.CATEGORIES))
    for col, (cat, column) in zip(cols, config.CATEGORIES.items()):
        r = rec["ratings"][cat]
        with col:
            st.markdown(f"**{column}**")
            st.markdown(f"### {MARK[r['severity']]} {r['severity']}")
            if gold:
                g = gold[cat]["severity"]
                st.caption(f"Parent-guide label: {MARK[g]} {g} {'✓' if g == r['severity'] else '✗'}")
            st.write(r["rationale"])
            for quote in r["evidence"]:
                quote = re.sub(r"<[^>]+>", "", quote)  # subtitle markup such as <i>
                st.markdown("> " + quote.replace("\n", "  \n> "))
    calls = rec.get("tool_calls") or []
    u = rec["usage"]
    st.caption(
        f"{llm.PROVIDERS[rec['provider']]['label']} {rec['model']} · effort {rec['effort']} · {u['calls']} API calls · "
        f"{u['input_tokens']:,} in / {u['output_tokens']:,} out tokens · ${u['cost_usd']:.4f} at list price · "
        f"{rec['seconds']}s · {len(calls)} tool calls"
    )
    if calls:
        with st.expander(f"Agent tool calls ({len(calls)})"):
            for c in calls:
                st.json(c, expanded=False)


def safe_error(exc: Exception, key: str) -> str:
    if isinstance(exc, openai.AuthenticationError):
        return "The provider rejected that API key."
    if isinstance(exc, openai.RateLimitError):
        return "The provider's rate limit was hit. Wait a minute, or try RAG mode (fewer tokens)."
    text = f"{type(exc).__name__}: {exc}"
    return text.replace(key, "[key]") if key else text


# Sidebar: provider, model and the visitor's key
with st.sidebar:
    st.header("Your model")
    provider = st.selectbox(
        "Provider", list(llm.PROVIDERS), format_func=lambda p: llm.PROVIDERS[p]["label"]
    )
    spec = llm.PROVIDERS[provider]
    model = st.selectbox("Model", spec["models"])
    effort = st.select_slider("Reasoning effort", ["low", "medium", "high"], value="medium")
    typed = st.text_input(
        f"{spec['label']} API key",
        type="password",
        help="Used for this run only. It is not stored, logged or sent anywhere except the provider.",
    )
    st.caption(f"[Get a {spec['label']} key]({spec['key_url']}). Groq has a free tier.")
    key = demo.resolve_key(provider, typed)
    if demo.DEV_ENV_KEYS and key and not typed:
        st.info("Dev mode: using the key from your local environment.")

st.title("🎬 How would a parent guide rate this film?")
st.write(
    "An LLM reads only the subtitles (the title is withheld from it) and rates violence, sex, language "
    "and substances as none / mild / moderate / strong, quoting the lines it relied on. "
    f"[How well it works on 75 films]({REPO}#results)."
)

source = st.radio("Subtitles", ["A film from the dataset", "Upload my own (.srt or .xml)"], horizontal=True)
mode = st.radio("Method", list(MODE_HELP), format_func=lambda m: MODE_HELP[m].split(":")[0], horizontal=True)
st.caption(MODE_HELP[mode])

film, upload = None, None
if source == "A film from the dataset":
    films = film_list()
    film = st.selectbox("Film", films, format_func=lambda f: f["review_title"].strip())
    gold = demo.review_labels(film["film_id"])
    saved = demo.precomputed(film["film_id"], mode)
    if saved:
        with st.expander(f"Saved {demo.DEMO_MODEL} rating ({mode}), no key needed", expanded=not key):
            show_rating(saved, gold)
else:
    gold = None
    upload = st.file_uploader("Subtitle file", type=["srt", "xml"])

ready = bool(key) and (film is not None or upload is not None)
if provider == "groq" and key:
    st.caption("Groq's free tier allows about 8K tokens a minute, so agentic runs can pause for a minute between steps.")
if st.button("Rate with my key", type="primary", disabled=not ready):
    ctx = None
    st.session_state.pop("last", None)
    try:
        backend = demo.make_backend(provider, model, key, effort)
        with st.status("Rating…", expanded=True) as status:
            if upload is not None:
                ctx = demo.upload_context(demo.parse_upload(upload.name, upload.getvalue()), store())
            else:
                st.write("Embedding the subtitles…")
                ctx = demo.dataset_context(film, store())

            def on_step(event):
                st.write(f"🔎 `{event['tool']}` {event['input']}")

            rec = rater.rate(ctx, mode, backend, on_step=on_step)
            status.update(label="Done", state="complete", expanded=False)
        # Keep the result (never the key) so it survives Streamlit's reruns.
        name = upload.name if upload is not None else film["review_title"].strip()
        st.session_state["last"] = {"name": name, "rec": rec, "gold": gold}
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:  # show provider errors without the key
        st.error(safe_error(exc, key))
    finally:
        if ctx is not None and upload is not None:
            demo.forget(ctx)
elif not key:
    st.caption("Paste an API key in the sidebar to run the rater yourself.")

if last := st.session_state.get("last"):
    st.subheader(f"Your run: {last['name']} ({last['rec']['mode']})")
    show_rating(last["rec"], last["gold"])
