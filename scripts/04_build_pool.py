#!/usr/bin/env python3
"""Step 4: build the blind judgment pool for the problem-level check.

Ten query papers are drawn from the TEST split of the May-2026 corpus, one per
primary category, taking the ten most frequent categories of the test split
(seed 11). For each query the top five of every system in
results/may2026/rankings_test_top100.npz are pooled. The pool is shuffled and
written without any system name or arXiv category, so the judge sees only the
query and candidate titles and abstracts.

Outputs (revision/judgments/):
  pool_blind.json  - what the judge sees
  pool_key.json    - which systems retrieved each candidate (kept away from the judge)
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT.parents[1]
CORPUS = ROOT / "data" / "arxiv_may2026.jsonl"            # repository layout
if not CORPUS.exists():                                    # original project layout
    CORPUS = PAPER_DIR / "experiments" / "data" / "raw" / "arxiv_papers.jsonl"
RANKS = ROOT / "results" / "may2026" / "rankings_test_top100.npz"
OUT = ROOT / "judgments"
N_QUERIES = 10
POOL_K = 5
SEED = 11


def main() -> None:
    papers = [json.loads(line) for line in open(CORPUS, encoding="utf-8") if line.strip()]
    z = np.load(RANKS)
    queries = z["queries"]
    systems = [k for k in z.files if k != "queries"]
    primary = [p["primary_category"] for p in papers]
    cats = [c for c, _ in Counter(primary[q] for q in queries).most_common(N_QUERIES)]
    rng = random.Random(SEED)
    chosen = []
    for c in cats:
        pos = [i for i, q in enumerate(queries) if primary[q] == c]
        chosen.append(rng.choice(pos))

    blind, key = [], {"systems": systems, "pool_k": POOL_K, "seed": SEED, "queries": []}
    for qn, pos in enumerate(chosen, 1):
        q = int(queries[pos])
        found: dict[int, list[str]] = {}
        for s in systems:
            for d in z[s][pos][:POOL_K]:
                found.setdefault(int(d), []).append(s)
        cands = list(found)
        rng.shuffle(cands)
        qid = f"Q{qn:02d}"
        blind.append({
            "qid": qid,
            "query": {"title": papers[q]["title"], "abstract": papers[q]["abstract"]},
            "candidates": [{"cid": f"{qid}-C{j+1:02d}", "title": papers[d]["title"],
                            "abstract": papers[d]["abstract"]} for j, d in enumerate(cands)],
        })
        key["queries"].append({
            "qid": qid, "row": q, "arxiv_id": papers[q]["arxiv_id"], "primary": primary[q],
            "candidates": [{"cid": f"{qid}-C{j+1:02d}", "row": d, "arxiv_id": papers[d]["arxiv_id"],
                            "primary": primary[d], "systems": found[d],
                            "rank": {s: int(np.flatnonzero(z[s][pos] == d)[0]) + 1
                                     for s in found[d]}} for j, d in enumerate(cands)],
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pool_blind.json").write_text(json.dumps(blind, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "pool_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = [len(b["candidates"]) for b in blind]
    print(f"{len(blind)} queries, {sum(sizes)} candidates, per query {sizes}")


if __name__ == "__main__":
    main()
