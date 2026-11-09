#!/usr/bin/env python3
"""Step 5: descriptive checks on both corpora and on the test rankings.

Collection facts (versions, duplicates, dates, language proxy, category mix)
and two list checks asked for by the reviewers: how often a top-5 hit shares an
author with the query, and how close its title is to the query title.
Writes revision/results/descriptives.json.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT.parents[1]
CORPORA = {
    # repository layout first, then the original project layout
    "may2026": (ROOT / "data" / "arxiv_may2026.jsonl") if (ROOT / "data" / "arxiv_may2026.jsonl").exists()
               else PAPER_DIR / "experiments" / "data" / "raw" / "arxiv_papers.jsonl",
    "y2020_2024": ROOT / "data" / "arxiv_2020_2024.jsonl",
}
WORD = re.compile(r"[a-z0-9]+")


def ascii_letter_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) / max(1, len(letters))


def title_jaccard(a: str, b: str) -> float:
    sa, sb = set(WORD.findall(a.lower())), set(WORD.findall(b.lower()))
    return len(sa & sb) / max(1, len(sa | sb))


def main() -> None:
    out = {}
    for name, path in CORPORA.items():
        papers = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        ids = [p["arxiv_id"] for p in papers]
        base = [i.rsplit("v", 1)[0] for i in ids]
        prim = Counter(p["primary_category"] for p in papers)
        words = [len(WORD.findall(p["abstract"])) for p in papers]
        d = {
            "n": len(papers),
            "unique_ids": len(set(ids)),
            "unique_base_ids": len(set(base)),
            "version_suffix": dict(Counter(i.rsplit("v", 1)[1] for i in ids)),
            "date_min": min(p["publication_date"] for p in papers),
            "date_max": max(p["publication_date"] for p in papers),
            "abstracts_below_90pct_ascii_letters": sum(ascii_letter_share(p["abstract"]) < 0.9 for p in papers),
            "n_primary_categories": len(prim),
            "primary_in_query_cats": sum(v for k, v in prim.items() if k in {"cs.AI", "cs.LG", "cs.CL", "cs.IR"}) / len(papers),
            "primary_top8": {k: v for k, v in prim.most_common(8)},
            "categories_per_paper_mean": float(np.mean([len(p.get("categories") or []) for p in papers])),
            "abstract_words_median": float(np.median(words)),
            "authors_per_paper_median": float(np.median([len(p.get("authors") or []) for p in papers])),
        }
        rk = ROOT / "results" / name / "rankings_test_top100.npz"
        if rk.exists():
            z = np.load(rk)
            q = z["queries"]
            authors = [set(a.strip().lower() for a in (p.get("authors") or [])) for p in papers]
            checks = {}
            for s in z.files:
                if s == "queries":
                    continue
                top = z[s][:, :5]
                shared = sum(bool(authors[qi] & authors[d]) for qi, row in zip(q, top) for d in row)
                tj = [title_jaccard(papers[qi]["title"], papers[d]["title"]) for qi, row in zip(q, top) for d in row]
                prim = np.array([p["primary_category"] for p in papers])
                base = np.mean([np.mean(prim[row[:5]] == prim[qi]) for qi, row in zip(q, z[s])])
                filt = np.mean([np.mean(prim[[d for d in row if not (authors[qi] & authors[d])][:5]] == prim[qi])
                                for qi, row in zip(q, z[s])])
                checks[s] = {"P@5": float(base), "P@5_without_shared_author": float(filt),
                             "slots": int(top.size), "shared_author_slots": int(shared),
                             "shared_author_share": shared / top.size,
                             "title_jaccard_ge_0.5": int(sum(t >= 0.5 for t in tj)),
                             "title_jaccard_max": float(max(tj))}
            d["top5_checks"] = checks
        out[name] = d
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "descriptives.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
