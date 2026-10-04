import math
import re
import unicodedata
from collections import Counter

# Groq free tier rejects requests above 8k tokens/minute; Vietnamese text measured ~0.32 token/char.
# System prompt + examples + schema + output reserve take ~3k tokens, questions up to ~1.3k.
CONTEXT_CHARS = 10_000
CHUNK_CHARS = 800
GAP = "\n[...]\n"
QUESTION = re.compile(r"\b(cau|bai|question|exercise)\s*\d+")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def terms(text: str) -> list[str]:
    """Accent-insensitive words plus bigrams, so 'cau 3' matches 'Câu 3'."""
    words = re.findall(r"\w+", fold(text))
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def split_line(line: str) -> list[str]:
    pieces = []
    while len(line) > CHUNK_CHARS:
        cut = line.rfind(" ", 0, CHUNK_CHARS)
        cut = cut if cut > 0 else CHUNK_CHARS
        pieces.append(line[:cut].strip())
        line = line[cut:].strip()
    return pieces + [line] if line else pieces


def chunk_pages(pages: list[str]) -> list[tuple[int, str]]:
    """Split by page, then by line; overlong lines are cut at word boundaries."""
    chunks = []
    for number, text in enumerate(pages, 1):
        current = ""
        for line in text.splitlines():
            for piece in split_line(line.strip()):
                if current and len(current) + len(piece) + 1 > CHUNK_CHARS:
                    chunks.append((number, current))
                    current = ""
                current = f"{current}\n{piece}" if current else piece
        if current:
            chunks.append((number, current))
    return chunks


def bm25(chunks: list[tuple[int, str]], query: str) -> list[float]:
    docs = [Counter(terms(text)) for _, text in chunks]
    wanted = set(terms(query))
    lengths = [sum(doc.values()) for doc in docs]
    average = sum(lengths) / len(docs) or 1
    frequency = Counter(term for doc in docs for term in wanted if term in doc)
    scores = []
    for doc, length in zip(docs, lengths):
        score = 0.0
        for term in wanted:
            if doc[term]:
                idf = math.log(1 + (len(docs) - frequency[term] + 0.5) / (frequency[term] + 0.5))
                score += idf * doc[term] * 2.5 / (doc[term] + 1.5 * (0.25 + 0.75 * length / average))
        scores.append(score)
    return scores


def select_context(pages: list[str], query: str, mode: str) -> tuple[list[dict], bool]:
    """Whole PDF when it fits the budget; otherwise the most relevant chunks, kept in page order."""
    if sum(map(len, pages)) <= CONTEXT_CHARS:
        return [{"page": i + 1, "text": text} for i, text in enumerate(pages)], False
    chunks = chunk_pages(pages)
    if mode == "solve":
        # Exercises in document order ("Câu 3", "Bài 2"), so the first questions are solved first.
        ranked = [i for i, (_, text) in enumerate(chunks) if QUESTION.search(fold(text))]
    else:
        # Explicit "trang N" in the question pins that page first.
        pinned = {int(n) for n in re.findall(r"\btrang\s*(\d+)", fold(query))}
        scores = [score + (1000 if page in pinned else 0) for score, (page, _) in zip(bm25(chunks, query), chunks)]
        ranked = sorted((i for i in range(len(chunks)) if scores[i] > 0), key=lambda i: -scores[i])
    chosen, used = [], 0
    for i in ranked or range(len(chunks)):
        if used + len(chunks[i][1]) <= CONTEXT_CHARS:
            chosen.append(i)
            used += len(chunks[i][1])
    selected = []
    for i in sorted(chosen):
        page, text = chunks[i]
        if selected and selected[-1]["page"] == page:
            selected[-1]["text"] += ("\n" if selected[-1]["last"] == i - 1 else GAP) + text
            selected[-1]["last"] = i
        else:
            selected.append({"page": page, "text": text, "last": i})
    return [{"page": item["page"], "text": item["text"]} for item in selected], True
