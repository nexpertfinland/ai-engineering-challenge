"""Original notebook pipeline, kept for an honest before/after comparison."""

import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from rag.llm import call_llm


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BaselinePipeline:
    def __init__(self, corpus: list[dict]):
        self.chunks = [
            {"doc_id": doc["doc_id"], "text": doc["text"][i:i + 500]}
            for doc in corpus for i in range(0, len(doc["text"]), 500)
        ]
        # Sparse storage changes memory use, not counts, scores, or ranking.
        self.vectorizer = CountVectorizer(analyzer=tokenize, dtype=np.float32)
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, question: str, k: int = 1) -> list[dict]:
        scores = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()
        return [self.chunks[i] for i in np.argsort(-scores)[:k]]

    def answer_question(self, question: str) -> dict:
        retrieved = self.retrieve(question, k=1)
        context = "\n\n".join(r["text"] for r in retrieved)
        prompt = (
            "Answer the question using only the context below. "
            "If the answer isn't in the context, say so briefly.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        answer = call_llm(prompt)
        return {"answer": answer.strip(), "citations": list({r["doc_id"] for r in retrieved})}
