import math
import re
import unicodedata
from collections import Counter

# Groq free tier rejects requests above 8k tokens/minute; Vietnamese text measured ~0.32 token/char.
# System prompt + examples + schema + output reserve take ~3k tokens, questions up to ~1.3k.
CONTEXT_CHARS = 8_000
CHUNK_CHARS = 800
GAP = "\n[...]\n"
QUESTION = re.compile(r"\b(cau|bai|question|exercise)\s*\d+")
SUMMARY_CUES = ("tom tat", "y chinh", "tong quan", "summary", "overview")
MODEL_LIST_CUE = re.compile(r"\b(?:models?|llms?)\s*(?::|include\b|are\b|such as\b|like\b)")
REFERENCE_LIST_CUE = re.compile(r"\breferences?\s*(?:\[\s*1\s*\]|1\.)")
SUMMARY_GENERIC_WORDS = {
    "a", "all", "an", "and", "are", "ca", "cac", "content", "document", "entire", "for", "in", "key",
    "main", "noi", "of", "on", "overview", "paper", "phan", "points", "summary", "takeaways", "tat",
    "the", "this", "to", "toan", "tong", "trang", "trong", "ve", "van", "y", "chinh", "lieu", "nguon",
    "dung", "bo", "kem", "tai",
}


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


def is_document_wide_summary(query: str) -> bool:
    normalized = fold(query)
    if not any(cue in normalized for cue in SUMMARY_CUES):
        return False
    topic = re.sub(r"\b(?:tom tat|y chinh|tong quan|summary|overview)\b", " ", normalized)
    terms = set(re.findall(r"\w+", topic))
    return not any(len(term) > 1 and term not in SUMMARY_GENERIC_WORDS for term in terms)


def is_model_list_question(query: str) -> bool:
    normalized = fold(query)
    asks_for_models = re.search(r"\b(?:which|what|list|name|identify)\b", normalized)
    mentions_models = re.search(r"\b(?:models?|llms?)\b", normalized)
    return bool(asks_for_models and mentions_models)


def is_first_reference_question(query: str) -> bool:
    normalized = fold(query)
    return bool(
        re.search(r"\b(?:first|1st|dau tien|thu nhat)\b", normalized)
        and re.search(r"\b(?:reference|citation|source|tham khao|nguon)\b", normalized)
    )


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
    """Budgeted retrieval with per-query coverage, neighboring context, and summary sampling."""
    if sum(map(len, pages)) <= CONTEXT_CHARS:
        return [{"page": i + 1, "text": text} for i, text in enumerate(pages)], False
    chunks = chunk_pages(pages)
    if not chunks:
        return [], True
    normalized = fold(query)
    document_wide_summary = is_document_wide_summary(query)
    pinned = {int(n) for n in re.findall(r"\b(?:trang|page)\s*(\d+)", normalized)}
    seeds = []
    scores = bm25(chunks, query)
    if mode == "solve":
        seeds = [i for i, (_, text) in enumerate(chunks) if QUESTION.search(fold(text))]
    elif document_wide_summary and not pinned:
        # Sample across the full document instead of silently reading only its beginning.
        page_first = list(dict.fromkeys(page for page, _ in chunks))
        capacity = max(1, CONTEXT_CHARS // (CHUNK_CHARS + len(GAP)))
        samples = min(capacity, len(page_first))
        positions = sorted({round(i * (len(page_first) - 1) / max(1, samples - 1)) for i in range(samples)})
        seeds = [next(i for i, (page, _) in enumerate(chunks) if page == page_first[pos]) for pos in positions]
    else:
        if not pinned and is_model_list_question(query):
            seeds.extend(i for i, (_, text) in enumerate(chunks) if MODEL_LIST_CUE.search(fold(text)))
        if not pinned and is_first_reference_question(query):
            seeds.extend(i for i, (_, text) in enumerate(chunks) if REFERENCE_LIST_CUE.search(fold(text)))
        if not seeds:
            # Cover each subquestion before letting a frequent topic monopolize the budget.
            parts = [part.strip() for part in re.split(r"[?;\n]+", query) if part.strip()][:12]
            for part in parts:
                part_scores = bm25(chunks, part)
                candidates = [i for i, (page, _) in enumerate(chunks) if not pinned or page in pinned]
                candidates = sorted(candidates, key=lambda i: -part_scores[i])
                if candidates and (part_scores[candidates[0]] > 0 or pinned):
                    seeds.append(candidates[0])
        seeds.extend(i for i, (page, _) in enumerate(chunks) if page in pinned)
    ranked = sorted((i for i in range(len(chunks)) if scores[i] > 0), key=lambda i: -scores[i])
    order = list(dict.fromkeys(seeds))
    # Include continuations of a selected exercise/paragraph on the same or next page.
    for i in list(order):
        for neighbor in (i + 1, i - 1):
            if 0 <= neighbor < len(chunks) and abs(chunks[neighbor][0] - chunks[i][0]) <= 1:
                if neighbor not in order:
                    order.append(neighbor)
    if mode != "solve" and not document_wide_summary:
        order.extend(i for i in ranked if i not in order)
    chosen, used = [], 0
    for i in order or range(len(chunks)):
        # Reserve separators too; the serialized text stays within the character budget.
        cost = len(chunks[i][1]) + len(GAP)
        if used + cost <= CONTEXT_CHARS:
            chosen.append(i)
            used += cost
    selected = []
    for i in sorted(chosen):
        page, text = chunks[i]
        if selected and selected[-1]["page"] == page:
            selected[-1]["text"] += ("\n" if selected[-1]["last"] == i - 1 else GAP) + text
            selected[-1]["last"] = i
        else:
            selected.append({"page": page, "text": text, "last": i})
    return [{"page": item["page"], "text": item["text"]} for item in selected], True
