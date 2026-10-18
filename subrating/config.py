"""Paths, categories, the severity rubric and run settings shared by every stage."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # API keys and optional overrides; never committed
SUBTITLE_DIR = ROOT / "Modelling_Dataset" / "Subtitles_XML"
REVIEW_DIR = ROOT / "Evaluation_Data" / "CS_Movie_Reviews"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"
CHROMA_DIR = CACHE_DIR / "chroma"

MATCHED_FILMS_CSV = DATA_DIR / "matched_films.csv"
LABELS_JSONL = RESULTS_DIR / "review_labels.jsonl"
LABELS_CHECK_JSONL = RESULTS_DIR / "review_labels_check.jsonl"
RATINGS_DIR = RESULTS_DIR / "ratings"


def ratings_dir(model: str, mode: str) -> Path:
    """results/ratings/<model>/<mode>/, so runs with different models never mix."""
    return RATINGS_DIR / model.replace("/", "_") / mode

# Any OpenAI-compatible provider works (see llm.PROVIDERS). The defaults are
# an open-weight model on Groq; override with env vars or CLI flags.
PROVIDER = os.environ.get("PROVIDER", "groq")
RATER_MODEL = os.environ.get("RATER_MODEL", "openai/gpt-oss-120b")
# Labels come from a different model family than the rater, and a second,
# independent labeler measures how noisy the labels are. On Groq each model
# has its own rate limit, so this also lets the steps run side by side.
LABEL_MODEL = os.environ.get("LABEL_MODEL", "qwen/qwen3.8-27b")
CHECK_LABEL_MODEL = os.environ.get("CHECK_LABEL_MODEL", "openai/gpt-oss-20b")
EFFORT = os.environ.get("EFFORT", "medium")  # reasoning effort: low | medium | high

SEVERITIES = ["none", "mild", "moderate", "strong"]

# The four review columns we evaluate. "Products & Purchases" is skipped: 60 of
# 83 reviews leave it empty and product placement is mostly visual.
CATEGORIES = {
    "violence": "Violence & Scariness",
    "sex": "Sex, Romance & Nudity",
    "language": "Language",
    "substances": "Drinking, Drugs & Smoking",
}

# One rubric, used both to label the reviews and to prompt the rater, so the
# two sides are scored on the same scale.
RUBRIC = {
    "violence": {
        "none": "No violence, peril or frightening content.",
        "mild": "Slapstick, cartoonish or fantasy peril; brief scuffles; mild suspense; no injuries shown.",
        "moderate": "Realistic fights, guns or weapons used, deaths that are not graphic, sustained tension or scary scenes.",
        "strong": "Graphic, bloody or brutal violence; torture; gore; intense horror; disturbing deaths.",
    },
    "sex": {
        "none": "No romance or sexual content.",
        "mild": "Kissing, flirting, crushes, very mild innuendo.",
        "moderate": "Sexual references or innuendo, crude sexual talk, implied sex, partial nudity.",
        "strong": "Sex scenes, graphic nudity, or frequent explicit sexual dialogue.",
    },
    "language": {
        "none": "No profanity or insults.",
        "mild": "Words like 'hell', 'damn', 'crap', 'stupid', name-calling.",
        "moderate": "Occasional stronger words ('s--t', 'b---h', 'a--hole') or crude language, without frequent f-words.",
        "strong": "Use of the f-word or slurs, or frequent strong profanity.",
    },
    "substances": {
        "none": "No drinking, smoking or drugs.",
        "mild": "Incidental or background drinking or smoking, a toast, passing references.",
        "moderate": "Main characters drink or smoke on screen regularly, drunkenness, references to drug use.",
        "strong": "Drug use or dealing shown, heavy drunkenness, addiction as a theme.",
    },
}

# Fixed queries used by the plain RAG baseline (and as the agent's first
# retrieval). Written as dialogue-like text because the index holds dialogue.
CATEGORY_QUERIES = {
    "violence": "I'm going to kill you. Gun, shoot, blood, dead body, fight, hurt, scream, run, monster, murder",
    "sex": "sex, sleep with me, naked, kiss, make love, sexy, bed, affair, breasts",
    "language": "fuck, shit, damn, hell, bitch, ass, bastard, swearing and insults",
    "substances": "drink, beer, whiskey, drunk, wine, cigarette, smoke, drugs, cocaine, weed, high",
}

# Retrieval settings. A chunk is CHUNK_SIZE consecutive subtitle sentences.
CHUNK_SIZE = 4
TOP_K = 12          # chunks per category for the RAG baseline (about 48 lines)
AGENT_SEARCH_K = 6  # chunks returned per agent search call
AGENT_MAX_SEARCHES = 8
# Stop agent searches once a request would pass this many (approximate)
# tokens. Unset: the provider's default (6500 on Groq for its free tier,
# no limit elsewhere). 0 disables it.
MAX_PROMPT_TOKENS = int(os.environ["MAX_PROMPT_TOKENS"]) if os.environ.get("MAX_PROMPT_TOKENS") else None
