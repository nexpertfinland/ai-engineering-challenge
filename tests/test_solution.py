import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rag.baseline import BaselinePipeline, tokenize as baseline_tokenize
from rag.llm import call_llm, get_usage, reset_usage, ensure_model_ready
from rag.pipeline import RAGPipeline, parse_answer
from rag.retrieval import RetrievalConfig, Retriever, chunk_corpus


CORPUS = [
    {"doc_id": "Solar", "title": "Solar System", "text": "Venus has a thick atmosphere. Its surface is very hot.\n\nNeptune has fourteen moons. Neptune is blue."},
    {"doc_id": "Ocean", "title": "Ocean", "text": "Blue whales live in the ocean. A whale is a mammal."},
]


class RetrievalTests(unittest.TestCase):
    def test_paragraphs_remain_separate_exact_source_slices(self):
        chunks = chunk_corpus(CORPUS, RetrievalConfig())
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            original = next(d for d in CORPUS if d["doc_id"] == chunk["doc_id"])
            self.assertIn(chunk["text"], original["text"])
            self.assertNotIn("\n\n", chunk["text"])

    def test_long_windows_cover_all_words_with_overlap_and_progress(self):
        text = " ".join(f"word{i}" + ("." if i % 11 == 10 else "") for i in range(97))
        chunks = chunk_corpus([{"doc_id": "d", "text": text}], RetrievalConfig(max_words=23, overlap_words=5))
        covered = set()
        for chunk in chunks:
            self.assertLessEqual(chunk["end_word"] - chunk["start_word"], 23)
            covered.update(range(chunk["start_word"], chunk["end_word"]))
            self.assertIn(chunk["text"], text)
        self.assertEqual(covered, set(range(97)))
        self.assertTrue(all(b["start_word"] > a["start_word"] for a, b in zip(chunks, chunks[1:])))
        self.assertTrue(all(b["start_word"] < a["end_word"] for a, b in zip(chunks, chunks[1:])))

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            RetrievalConfig(max_words=10, overlap_words=10)

    def test_relevant_paragraph_ranks_first_and_ranking_is_repeatable(self):
        retriever = Retriever(CORPUS)
        hits = retriever.retrieve("How many moons does Neptune have?", 3)
        self.assertIn("fourteen", hits[0]["text"])
        self.assertEqual(hits, retriever.retrieve("How many moons does Neptune have?", 3))

    def test_unknown_or_blank_query_and_zero_k_return_no_evidence(self):
        retriever = Retriever(CORPUS)
        for question, k in [("xyzzzzz", 3), ("", 3), ("Neptune", 0)]:
            self.assertEqual(retriever.retrieve(question, k), [])

    def test_empty_corpus_fails_clearly(self):
        with self.assertRaises(ValueError):
            Retriever([])

    def test_sparse_baseline_matches_original_dense_scores_and_order(self):
        baseline = BaselinePipeline(CORPUS)
        vocab = {}
        for chunk in baseline.chunks:
            for token in baseline_tokenize(chunk["text"]):
                vocab.setdefault(token, len(vocab))
        def vector(text):
            result = np.zeros(len(vocab), dtype=np.float32)
            for token in baseline_tokenize(text):
                if token in vocab:
                    result[vocab[token]] += 1
            return result
        matrix = np.stack([vector(c["text"]) for c in baseline.chunks])
        for question in ["How many moons does Neptune have?", "the ocean", "unknown"]:
            indices = np.argsort(-(matrix @ vector(question)))[:3]
            self.assertEqual(baseline.retrieve(question, 3), [baseline.chunks[i] for i in indices])


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.passages = [{"doc_id": "one", "text": "The answer is fourteen moons."}, {"doc_id": "two", "text": "Blue whales are mammals."}]

    def test_only_the_answer_supporting_source_is_cited(self):
        self.assertEqual(parse_answer('{"answer":"fourteen moons","source":1}', self.passages),
                         {"answer": "fourteen moons", "citations": ["one"]})

    def test_non_grounded_or_wrong_source_answer_is_rejected(self):
        for raw in ['{"answer":"fifteen","source":1}', '{"answer":"fourteen","source":2}',
                    '{"answer":"fourteen","source":99}', '{"answer":"","source":0}']:
            self.assertEqual(parse_answer(raw, self.passages), {"answer": "", "citations": []})

    def test_malformed_or_untyped_outputs_are_rejected(self):
        for raw in ['not json', '[]', 'null', '{"answer":14,"source":1}',
                    '{"answer":"fourteen","source":true}', '{"answer":"fourteen","source":"1"}']:
            self.assertEqual(parse_answer(raw, self.passages), {"answer": "", "citations": []})

    def test_code_fences_case_and_whitespace_are_tolerated(self):
        raw = '```json\n{"answer":"Fourteen  moons","source":1}\n```'
        self.assertEqual(parse_answer(raw, self.passages)["citations"], ["one"])

    @patch("rag.pipeline.call_llm", return_value='{"answer":"fourteen","source":1}')
    def test_pipeline_uses_one_short_structured_call(self, llm):
        pipeline = RAGPipeline(CORPUS)
        result = pipeline.answer_question("How many moons does Neptune have?")
        self.assertEqual(result, {"answer": "fourteen", "citations": ["Solar"]})
        llm.assert_called_once()
        self.assertTrue(llm.call_args.kwargs["json_mode"])
        self.assertEqual(llm.call_args.kwargs["temperature"], 0)

    @patch("rag.pipeline.call_llm")
    def test_no_retrieval_evidence_means_no_model_call(self, llm):
        self.assertEqual(RAGPipeline(CORPUS).answer_question("xyzzzzz"), {"answer": "", "citations": []})
        llm.assert_not_called()


class ProviderTests(unittest.TestCase):
    def setUp(self):
        reset_usage()

    @patch("rag.llm.PROVIDER", "ollama")
    @patch("rag.llm.requests.post")
    def test_real_api_token_fields_are_recorded_for_every_call(self, post):
        response = Mock()
        response.json.return_value = {"message": {"content": '{"answer":"x","source":1}'},
                                      "prompt_eval_count": 31, "eval_count": 7}
        post.return_value = response
        call_llm("question", temperature=0, max_tokens=128, json_mode=True)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["options"]["num_predict"], 128)
        self.assertEqual(get_usage(), {"input_tokens": 31, "output_tokens": 7, "calls": 1})
        call_llm("baseline")
        self.assertNotIn("format", post.call_args.kwargs["json"])
        self.assertNotIn("options", post.call_args.kwargs["json"])
        self.assertEqual(get_usage()["calls"], 2)
        reset_usage()
        self.assertEqual(get_usage()["input_tokens"], 0)

    @patch("rag.llm.PROVIDER", "ollama")
    @patch("rag.llm.requests.get")
    def test_preflight_reports_missing_model(self, get):
        get.return_value.json.return_value = {"models": []}
        with self.assertRaisesRegex(RuntimeError, "ollama pull"):
            ensure_model_ready()


if __name__ == "__main__":
    unittest.main()
