"""Development-only diagnostics. The official scorer remains in eval/."""

import time

from eval.metrics import exact_match_score, f1_score
from rag.llm import get_usage, reset_usage


def retrieval_report(retrieve, dev_qa: list[dict], k: int) -> dict:
    if not dev_qa:
        raise ValueError("Development set is empty")
    doc_hits = answer_hits = chars = 0
    for item in dev_qa:
        passages = retrieve(item["question"], k=k)
        relevant = [p for p in passages if p["doc_id"] == item["doc_id"]]
        doc_hits += bool(relevant)
        answer_hits += any(
            " ".join(answer.casefold().split()) in " ".join(p["text"].casefold().split())
            for p in relevant for answer in item["answers"]
        )
        chars += sum(len(p["text"]) for p in passages)
    n = len(dev_qa)
    return {
        "n_questions": n, "k": k, "document_hits": doc_hits,
        "answer_context_hits": answer_hits,
        "document_recall": round(doc_hits / n, 4),
        "answer_context_recall": round(answer_hits / n, 4),
        "avg_context_characters": round(chars / n, 1),
    }


def evaluate_dev(answer_fn, dev_qa: list[dict], verbose: bool = True) -> dict:
    if not dev_qa:
        raise ValueError("Development set is empty")
    reset_usage()
    start = time.perf_counter()
    details = []
    for i, item in enumerate(dev_qa, 1):
        result = answer_fn(item["question"])
        details.append({
            "id": item["id"], "question": item["question"],
            "answer": result["answer"], "citations": result["citations"],
            "f1": f1_score(result["answer"], item["answers"]),
            "exact_match": exact_match_score(result["answer"], item["answers"]),
            "citation_hit": item["doc_id"] in result["citations"],
        })
        if verbose:
            print(f"dev {i}/{len(dev_qa)}: F1={details[-1]['f1']:.3f}", flush=True)
    n = len(details)
    f1 = sum(d["f1"] for d in details) / n
    citation_hit_rate = sum(d["citation_hit"] for d in details) / n
    usage = get_usage()
    return {
        "n_questions": n, "f1": round(f1, 4),
        "exact_match": round(sum(d["exact_match"] for d in details) / n, 4),
        "citation_hit_rate": round(citation_hit_rate, 4),
        "dev_score": round(100 * (0.8 * f1 + 0.2 * citation_hit_rate), 2),
        "total_time_sec": round(time.perf_counter() - start, 2),
        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        **usage, "details": details,
    }
