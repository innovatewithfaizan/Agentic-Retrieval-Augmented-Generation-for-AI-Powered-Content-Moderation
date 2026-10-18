"""Parse OpenSubtitles-style XML into clean lines and retrieval chunks."""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class Line:
    sid: str
    start: str  # "HH:MM:SS" of the first timestamp in the sentence, or ""
    text: str


@dataclass
class Chunk:
    chunk_id: str
    start: str
    text: str  # newline-joined lines


_FIXES = [
    (re.compile(r"(\w)' (s|t|re|ve|ll|d|m)\b", re.I), r"\1'\2"),  # "I' ve" -> "I've"
    (re.compile(r"(\w)' il\b", re.I), r"\1'll"),                    # OCR "I' il" -> "I'll"
    (re.compile(r"\s+([.,!?;:])"), r"\1"),                          # "word ." -> "word."
    (re.compile(r"\s+"), " "),
]


def clean(text: str) -> str:
    for pattern, repl in _FIXES:
        text = pattern.sub(repl, text)
    return text.strip()


def parse_xml(path: Path) -> list[Line]:
    root = ET.parse(path).getroot()
    lines = []
    last_start = "00:00:00"
    for s in root.iter("s"):
        words = [w.text or "" for w in s.iter("w")]
        text = clean(" ".join(words))
        if not text or re.fullmatch(r"[\d\s.,:\-]+", text):
            continue  # skip empty lines and stray frame counters like "23. 976"
        # Sentences that continue a subtitle frame have no start time of their
        # own; they inherit the previous one.
        for t in s.iter("time"):
            if t.get("id", "").endswith("S"):
                last_start = t.get("value", "")[:8]
                break
        lines.append(Line(s.get("id", ""), last_start, text))
    return lines


_SRT_TIME = re.compile(r"(\d{1,2}):(\d\d):(\d\d)[,.]\d+\s*-->")
_TAGS = re.compile(r"<[^>]+>|\{\\[^}]*\}")  # <i>, <font ...>, {\an8}


def parse_srt(text: str) -> list[Line]:
    """Parse an .srt file's text: one Line per subtitle block."""
    lines = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")):
        rows = [r.strip() for r in block.strip().split("\n")]
        times = [i for i, r in enumerate(rows) if _SRT_TIME.match(r)]
        if not times:
            continue
        h, m, s = _SRT_TIME.match(rows[times[0]]).groups()
        body = clean(_TAGS.sub("", " ".join(rows[times[0] + 1 :])))
        if body:
            lines.append(Line(str(len(lines) + 1), f"{int(h):02d}:{m}:{s}", body))
    return lines


def chunk_lines(lines: list[Line], size: int = config.CHUNK_SIZE) -> list[Chunk]:
    chunks = []
    for i in range(0, len(lines), size):
        group = lines[i : i + size]
        chunks.append(
            Chunk(
                chunk_id=f"c{i // size:04d}",
                start=group[0].start,
                text="\n".join(line.text for line in group),
            )
        )
    return chunks


def full_transcript(lines: list[Line]) -> str:
    return "\n".join(line.text for line in lines)
