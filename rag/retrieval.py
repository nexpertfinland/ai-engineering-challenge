"""Paragraph-aware sparse retrieval, fitted on the corpus only."""

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer


STOP_WORDS = ENGLISH_STOP_WORDS - {"no", "nor", "not"}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP_WORDS]


@dataclass(frozen=True)
class RetrievalConfig:
    max_words: int = 220
    overlap_words: int = 40
    bm25_weight: float = 0.6
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self):
        if not 0 <= self.overlap_words < self.max_words:
            raise ValueError("Require 0 <= overlap_words < max_words")
        if not 0 <= self.bm25_weight <= 1 or self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("Invalid retrieval weights")


def chunk_corpus(corpus: list[dict], config: RetrievalConfig) -> list[dict]:
    """Keep normal paragraphs whole; split long ones at nearby sentence boundaries.

    Every passage remains an exact slice of the source. Long windows overlap so
    answers near a boundary retain context. No question or answer labels are used.
    """
    chunks = []
    for doc in corpus:
        for paragraph_id, paragraph in enumerate(re.split(r"\n\s*\n", doc["text"])):
            words = list(re.finditer(r"\S+", paragraph))
            start = 0
            while start < len(words):
                end = min(start + config.max_words, len(words))
                if end < len(words):
                    lower = max(start + config.overlap_words + 1, end - 45)
                    for candidate in range(end, lower, -1):
                        if re.search(r'[.!?][\)\]"\u201d\u2019]*$', words[candidate - 1].group()):
                            end = candidate
                            break
                text = paragraph[words[start].start():words[end - 1].end()]
                chunks.append({
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", doc["doc_id"]).replace("_", " "),
                    "text": text,
                    "paragraph_id": paragraph_id,
                    "start_word": start,
                    "end_word": end,
                })
                if end == len(words):
                    break
                start = end - config.overlap_words
    return chunks


class Retriever:
    def __init__(self, corpus: list[dict], config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self.chunks = chunk_corpus(corpus, self.config)
        if not self.chunks:
            raise ValueError("Corpus has no text")
        texts = [f"{c['title']}\n{c['text']}" for c in self.chunks]
        self.counts = CountVectorizer(analyzer=tokenize, dtype=np.float32)
        counts = self.counts.fit_transform(texts).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf = np.log1p((len(texts) - df + 0.5) / (df + 0.5))
        norm = self.config.k1 * (1 - self.config.b + self.config.b * lengths / max(lengths.mean(), 1))
        self.bm25 = counts.copy()
        row_ids = np.repeat(np.arange(len(texts)), np.diff(counts.indptr))
        self.bm25.data = (
            counts.data * (self.config.k1 + 1) / (counts.data + norm[row_ids])
            * self.idf[counts.indices]
        )
        self.tfidf = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), sublinear_tf=True, norm="l2", dtype=np.float32,
        )
        self.tfidf_matrix = self.tfidf.fit_transform(texts)

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        if k <= 0 or not question.strip():
            return []
        query = self.counts.transform([question])
        query.data[:] = 1  # Repetition in a question should not inflate its score.
        bm25 = (self.bm25 @ query.T).toarray().ravel()
        tfidf = (self.tfidf_matrix @ self.tfidf.transform([question]).T).toarray().ravel()
        weight = self.config.bm25_weight
        scores = weight * bm25 / max(float(bm25.max()), 1e-12) + (1 - weight) * tfidf / max(float(tfidf.max()), 1e-12)
        order = np.argsort(-scores, kind="stable")
        selected = []
        for i in order:
            if scores[i] <= 0:
                break
            chunk = self.chunks[i]
            # Avoid spending the context budget twice on overlapping windows.
            if any(
                c["doc_id"] == chunk["doc_id"] and c["paragraph_id"] == chunk["paragraph_id"]
                and max(0, min(c["end_word"], chunk["end_word"]) - max(c["start_word"], chunk["start_word"]))
                / min(c["end_word"] - c["start_word"], chunk["end_word"] - chunk["start_word"]) > 0.5
                for c in selected
            ):
                continue
            selected.append({**chunk, "score": float(scores[i])})
            if len(selected) == k:
                break
        return selected
