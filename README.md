# arxiv-similar-paper-retrieval

Code and data for the paper **"Comparative Analysis of Vectorization Models and Retrieval Methods for Finding Similar Scientific Papers"** (A. Tursynbayev, Zh. Sadirmekova; ICTACS 2026, Tashkent).

The task is query-by-document retrieval: given one arXiv paper (title + abstract), rank the other papers of the corpus. The query paper is removed from its own candidate list. A retrieved paper counts as relevant when it shares the query's primary arXiv category.

Systems: BM25, TF-IDF cosine, LSA, `all-MiniLM-L6-v2`, SPECTER2 (proximity adapter), `bge-small-en-v1.5`, and a convex dense + BM25 hybrid.

## Main result (test split, P@5)

| Method | May 2026 | 2020–2024 |
|---|---|---|
| BM25 | 0.456 | 0.481 |
| TF-IDF | 0.429 | 0.452 |
| LSA (k=128) | 0.454 | 0.467 |
| MiniLM | 0.510 | 0.561 |
| SPECTER2 | 0.520 | 0.567 |
| BGE-small | 0.512 | 0.542 |
| Hybrid MiniLM+BM25 (w=0.6) | 0.509 | 0.548 |

8,368 and 8,497 test queries; 1,000 further queries per corpus are used only for parameter selection.

## Contents

| Path | What it is |
|---|---|
| `data/arxiv_may2026.jsonl` | 10,000 records, 5–28 May 2026, collected 31 May 2026. |
| `data/arxiv_2020_2024.jsonl` | 10,000 records sampled across 2020–2024 (4 windows per month, 45 papers each, seeded draw). |
| `scripts/00_check_env.py` | Checks Python, packages and access to arXiv and Hugging Face. |
| `scripts/01_collect_2020_2024.py` | Collects the 2020–2024 corpus from the arXiv API. |
| `scripts/02_encode.py` | Encodes a corpus with the dense models; one `.npy` per model in `embeddings/`. |
| `scripts/03_evaluate.py` | All retrieval runs, dev/test split, metrics, paired statistics, sensitivity grids, field ablation, hybrid weight curves. |
| `scripts/05_descriptives.py` | Corpus checks: versions, duplicates, dates, language proxy, shared authors and near-duplicate titles in the top 5. |
| `scripts/06_make_tables.py` | Writes every number, table and figure used in the paper. |
| `results/<corpus>/` | `results.json`, per-query test scores, top-100 rankings, dev query list. |
| `paper/` | LaTeX source and PDF of the camera-ready version. |

`scripts/04_build_pool.py` builds a blind judgment pool from the rankings for future problem-level relevance judgments.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/02_encode.py --corpus may2026
python scripts/02_encode.py --corpus y2020_2024
python scripts/03_evaluate.py --corpus may2026
python scripts/03_evaluate.py --corpus y2020_2024
python scripts/05_descriptives.py
python scripts/06_make_tables.py
```

Encoding SPECTER2 on a laptop CPU takes about 1.5 hours per corpus; the other models take minutes. Evaluation needs only NumPy, pandas, SciPy and scikit-learn and runs in about six minutes per corpus.

## License

Code: MIT. arXiv metadata: CC0 1.0, as distributed by arXiv.
