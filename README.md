# AI Engineering Challenge — Local RAG Question Answering

## Valmis workshop-ratkaisu

Tässä repossa on myös toteutettu ratkaisu: kappalerajat säilyttävä tiedonhaku,
BM25:n ja TF-IDF:n yhdistelmä sekä lyhyet, lähdetekstiin tarkistetut vastaukset.
Notebookin ohjeet ovat suomeksi. Alkuperäinen vertailuratkaisu on säilytetty
tiedostossa `rag/baseline.py` ja lukittu arviointikoodi on ennallaan.

**[Muutokset, mitatut tulokset ja ajo-ohjeet](WORKSHOP_RESULTS.md)**

Käynnistys Windowsissa, kun asennus on tehty ja Ollama on käynnissä:

```powershell
uv run python -m jupyterlab notebooks/challenge.ipynb
```

Tämä käynnistää JupyterLabin Python-moduulina. PowerShell-ikkuna jää auki
palvelimen ajaksi. Avaa notebook ja valitse **Run → Run All Cells**.
Jos virustorjunta näyttää uuden eston, selvitä se ennen jatkamista.

Alla on alkuperäinen tehtävänanto.

---

Answer questions against a local Wikipedia-derived corpus. The starter pipeline already
runs end-to-end and produces answers — just badly. Improve it, submit early, keep
iterating. Best score when time's up wins.

Score is computed by a **locked harness** (`eval/`) against a held-out question set —
the same one for every participant, so results are comparable. Anything outside `eval/`
is yours to change.

## Prerequisites

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) installed
    - For MacOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - For Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - Note that you might need to restart your terminal for uv install location to be added to PATH
- [Ollama](https://ollama.com/download) installed, running, with a model pulled:

  ```bash
  ollama pull llama3.2:3b
  ollama serve   # if it isn't already running as a background service
  ```

  Everything defaults to this local model — nothing leaves your laptop unless you
  deliberately switch providers (see below).

## Setup (~5 minutes)

```bash
git clone <repo-url>
cd ai-engineering-challenge
uv sync
uv run jupyter lab notebooks/challenge.ipynb
```

Run every cell top to bottom. The first cell builds the corpus from SQuAD the first
time you run it and caches it under `data/` for next time. The last cell prints a
`score` — that's your baseline result. Once you see it, flag it (green post-it on
your laptop) and start iterating.

## What you're optimizing

The pipeline has two stages, both fair game:

1. **Retrieval** — the baseline chunks articles into crude fixed-size windows and ranks
   them with raw word-count dot products (no normalization, no semantics).
2. **Generation** — the baseline retrieves exactly one chunk and asks the local model to
   answer from it with a bare-bones prompt.

The `## Ideas to try` cell at the bottom of the notebook lists concrete directions
(better vectorization, chunking, top-k + reranking, dense embeddings, prompt changes,
swapping models/providers). None of it is required reading — dig in wherever you want.

## Rules

- ✅ Edit the notebook and anything in `rag/` freely.
- ✅ Swap in any model — stick with the local default, try a different local model, or
  point at a paid/cloud API on your own key and your own dime (see below).
- ❌ Do not edit `eval/` — that's the scorer. Editing it to inflate your score is against
  the rules and will be checked.
- Re-run the last notebook cell as often as you like — there's no submission limit.

## Switching models / providers

Copy `.env.example` to `.env`. To use a different local Ollama model, set `OLLAMA_MODEL`.
To use a cloud model instead, set `LLM_PROVIDER=openai` or `LLM_PROVIDER=anthropic` plus
the matching API key — no code changes needed, `rag/llm.py` handles both.

## How scoring works

`eval/harness.py` calls your `answer_question(question)` function against 80 held-out
questions and reports **two independent scores** — there's no single winner metric:

- **`score` (higher is better)** — answer quality:
  - `f1` — token-overlap F1 between your answer and the gold answer (standard SQuAD metric)
  - `citation_hit_rate` — did you cite the article the answer actually came from
  - `score = 100 * (0.8 * f1 + 0.2 * citation_hit_rate)`
- **`total_tokens` (lower is more frugal)** — total input+output tokens spent answering
  all 80 questions, tracked automatically inside `rag/llm.py`'s `call_llm()`. This is a
  separate leaderboard: most accurate vs. most frugal don't have to be the same submission.

It also reports `total_time_sec` / `avg_latency_sec` per run, since latency vs. quality is
part of the local-model tradeoff this challenge is meant to surface.

Token tracking only counts calls made through `call_llm()`. If you rewrite `rag/llm.py`
from scratch, keep calling `_record_usage(input_tokens, output_tokens)` (or `total_tokens`
will just read `None` for you — not a crash, but you'll be opted out of the frugality score).

## Dataset & licensing

Corpus and questions are derived from [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/)
(Rajpurkar et al.), licensed **CC BY-SA 4.0**. Nothing derived from it is committed to
the repo: the notebook's setup cell builds `data/corpus.json`, `data/dev_qa.json`, and
`eval/test_qa.json` itself from SQuAD on first run (via `rag/dataset.py`) and caches
them locally. The split is deterministic (fixed seed), so every participant's held-out 
`eval/test_qa.json` comes out byte-identical without ever being shipped in the repo.

## Disclaimer

This workshop was built with the help of Claude 🤖
