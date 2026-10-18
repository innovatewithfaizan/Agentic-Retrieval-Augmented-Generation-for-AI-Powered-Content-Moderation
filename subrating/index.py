"""Vector store over subtitle chunks (ChromaDB + all-MiniLM-L6-v2) and the two
search functions the rater uses: semantic chunk search and exact word lookup."""

import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache

import chromadb
from chromadb.utils import embedding_functions

from . import config
from .subtitles import Line, chunk_lines, parse_xml

COLLECTION = "subtitle_chunks"


class Store:
    """One Chroma collection holding chunks of many films, keyed by film_id."""

    def __init__(self, client):
        # Chroma's default embedder is the ONNX export of all-MiniLM-L6-v2, the
        # same model the original notebook loaded through sentence-transformers.
        self.col = client.get_or_create_collection(
            COLLECTION,
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        self._lock = threading.Lock()

    def ensure(self, film_id: str, lines: list[Line]) -> int:
        """Embed a film's chunks unless they are already stored. Returns chunks added."""
        with self._lock:
            if self.col.get(where={"film_id": film_id}, limit=1)["ids"]:
                return 0
            chunks = chunk_lines(lines)
            self.col.add(
                ids=[f"{film_id}:{c.chunk_id}" for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[{"film_id": film_id, "start": c.start, "chunk": c.chunk_id} for c in chunks],
            )
            return len(chunks)

    def search(self, film_id: str, query: str, k: int, exclude=frozenset()) -> list[dict]:
        """Top-k chunks of one film by cosine similarity, skipping ids in exclude."""
        with self._lock:
            res = self.col.query(query_texts=[query], n_results=k + len(exclude), where={"film_id": film_id})
        hits = []
        for cid, doc, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0]):
            if cid in exclude:
                continue
            hits.append({"id": cid, "start": meta["start"], "text": doc})
            if len(hits) == k:
                break
        return sorted(hits, key=lambda h: h["start"])


@lru_cache(maxsize=1)
def persistent_store() -> Store:
    return Store(chromadb.PersistentClient(path=str(config.CHROMA_DIR)))


def memory_store() -> Store:
    return Store(chromadb.EphemeralClient())


@lru_cache(maxsize=None)
def film_lines(subtitle_file: str) -> list[Line]:
    return parse_xml(config.SUBTITLE_DIR / subtitle_file)


@dataclass
class FilmContext:
    """Everything the rater needs about one film. No title: it is withheld."""

    film_id: str
    lines: list[Line]
    store: Store
    seen: set = field(default_factory=set)  # chunk ids already shown to the model

    @classmethod
    def from_dataset(cls, film: dict, store: Store | None = None) -> "FilmContext":
        store = store or persistent_store()
        lines = film_lines(film["subtitle_file"])
        store.ensure(film["film_id"], lines)
        return cls(film["film_id"], lines, store)


def build_index(films: list[dict]) -> int:
    store = persistent_store()
    return sum(store.ensure(f["film_id"], film_lines(f["subtitle_file"])) for f in films)


# Common English endings, so "fuck" also finds "fucking" and "drunk" finds
# "drunken", without letting "hell" match "hello".
INFLECTIONS = r"(?:s|es|ed|d|ing|in|er|ers|y|en)?"


def find_words(lines: list[Line], words: list[str], max_examples: int = 8) -> dict:
    """Case-insensitive whole-word counts over every line of a film.

    Each word also matches common endings ("fuck" -> "fucking", "fuckin",
    "fucker"); a trailing '*' matches any ending."""
    counts, examples = {}, []
    for word in words[:12]:
        stem = re.escape(word.strip().rstrip("*"))
        if not stem:
            continue
        tail = r"\w*" if word.strip().endswith("*") else INFLECTIONS
        pattern = re.compile(rf"\b{stem}{tail}\b", re.I)
        n = 0
        for line in lines:
            hits = len(pattern.findall(line.text))
            if hits:
                n += hits
                entry = f"[{line.start}] {line.text}"
                if len(examples) < max_examples and entry not in examples:
                    examples.append(entry)
        counts[word] = n
    return {"counts": counts, "example_lines": examples}
