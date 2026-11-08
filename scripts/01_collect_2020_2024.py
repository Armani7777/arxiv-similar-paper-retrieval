#!/usr/bin/env python3
"""Step 1: collect a multi-year arXiv corpus (2020-2024) for the revision of paper 703.

Same category query and the same record fields as the original May-2026 corpus
(experiments/scripts/phase1_collect_arxiv.py), so the two corpora can be scored
by one evaluation script.

Sampling design (documented in the paper):
  * every month of 2020-2024 is split into four windows that start on days
    1, 8, 15 and 22 (the last window runs to the end of the month);
  * from each window the first PER_WINDOW papers by submission date are taken
    (arXiv API, sorted ascending by submittedDate);
  * duplicates (same base id) and records without abstract, date or primary
    category are dropped;
  * exactly TARGET records are then kept by a seeded uniform draw.

The script is resumable: finished windows are recorded in data/collect_state.json,
so after an interruption just run it again.
"""
from __future__ import annotations

import argparse
import calendar
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import arxiv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = ROOT / "logs"
RAW = DATA / "arxiv_2020_2024_raw.jsonl"
STATE = DATA / "collect_state.json"
FINAL = DATA / "arxiv_2020_2024.jsonl"
META = DATA / "arxiv_2020_2024_meta.json"
LOGFILE = LOGS / "01_collect.log"

CAT_QUERY = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.IR)"
WINDOW_STARTS = (1, 8, 15, 22)
YEARS = (2020, 2024)
PER_WINDOW = 45
TARGET = 10_000
SEED = 2020
DELAY_S = 3.0


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def windows():
    for y in range(YEARS[0], YEARS[1] + 1):
        for m in range(1, 13):
            last = calendar.monthrange(y, m)[1]
            for i, d0 in enumerate(WINDOW_STARTS):
                d1 = WINDOW_STARTS[i + 1] - 1 if i + 1 < len(WINDOW_STARTS) else last
                yield f"{y}{m:02d}{d0:02d}", f"{y}{m:02d}{d1:02d}"


def entry_to_record(entry: arxiv.Result, window: str) -> dict:
    # Identical to phase1_collect_arxiv.entry_to_record, plus all_categories and window.
    cats = [c for c in entry.categories if c.startswith("cs.")]
    return {
        "arxiv_id": entry.entry_id.split("/abs/")[-1],
        "title": entry.title.replace("\n", " ").strip(),
        "abstract": entry.summary.replace("\n", " ").strip(),
        "authors": [a.name for a in entry.authors],
        "categories": cats or list(entry.categories),
        "all_categories": list(entry.categories),
        "primary_category": entry.primary_category,
        "publication_date": entry.published.strftime("%Y-%m-%d"),
        "updated_date": entry.updated.strftime("%Y-%m-%d"),
        "pdf_url": entry.pdf_url,
        "window_start": window,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"done": {}, "failed": {}}


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def fetch_window(client: arxiv.Client, a: str, b: str) -> list[dict]:
    query = f"{CAT_QUERY} AND submittedDate:[{a}0000 TO {b}2359]"
    search = arxiv.Search(
        query=query,
        max_results=PER_WINDOW,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )
    return [entry_to_record(e, a) for e in client.results(search)]


def collect() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    state = load_state()
    client = arxiv.Client(page_size=PER_WINDOW, delay_seconds=DELAY_S, num_retries=5)
    all_w = list(windows())
    todo = [w for w in all_w if w[0] not in state["done"]]
    log(f"windows total={len(all_w)} done={len(all_w) - len(todo)} todo={len(todo)}")
    t0 = time.perf_counter()
    for n, (a, b) in enumerate(todo, 1):
        recs: list[dict] = []
        err = None
        for attempt in range(1, 4):
            try:
                recs = fetch_window(client, a, b)
                if recs:
                    break
                err = "empty result"
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
            log(f"  window {a}-{b} attempt {attempt} failed ({err}); sleeping 15 s")
            time.sleep(15)
        if recs:
            with open(RAW, "a", encoding="utf-8") as out:
                for r in recs:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
            state["done"][a] = len(recs)
            state["failed"].pop(a, None)
        else:
            state["failed"][a] = err
        save_state(state)
        if n % 10 == 0 or n == len(todo):
            el = time.perf_counter() - t0
            eta = el / n * (len(todo) - n)
            log(f"{n}/{len(todo)} windows, last {a}: {len(recs)} records, elapsed {el/60:.1f} min, ETA {eta/60:.1f} min")
        time.sleep(DELAY_S)
    if state["failed"]:
        log(f"FAILED windows: {sorted(state['failed'])}. Run the script again to retry them.")


def base_id(arxiv_id: str) -> str:
    return arxiv_id.rsplit("v", 1)[0] if "v" in arxiv_id else arxiv_id


def finalize() -> None:
    state = load_state()
    rows: list[dict] = []
    seen: set[str] = set()
    dropped = Counter()
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            b = base_id(r["arxiv_id"])
            if b in seen:
                dropped["duplicate"] += 1
                continue
            if not r.get("abstract") or not r.get("title") or not r.get("primary_category") or not r.get("publication_date"):
                dropped["missing_field"] += 1
                continue
            year = int(r["publication_date"][:4])
            if not (YEARS[0] <= year <= YEARS[1]):
                dropped["outside_years"] += 1
                continue
            seen.add(b)
            rows.append(r)
    candidates = len(rows)
    if len(rows) > TARGET:
        keep = set(random.Random(SEED).sample(range(len(rows)), TARGET))
        rows = [r for i, r in enumerate(rows) if i in keep]
    with open(FINAL, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "query": CAT_QUERY,
        "years": list(YEARS),
        "window_starts": list(WINDOW_STARTS),
        "per_window": PER_WINDOW,
        "windows_total": len(list(windows())),
        "windows_done": len(state["done"]),
        "windows_failed": state["failed"],
        "candidates_after_cleaning": candidates,
        "dropped": dict(dropped),
        "final_n": len(rows),
        "seed": SEED,
        "per_year": dict(sorted(Counter(r["publication_date"][:4] for r in rows).items())),
        "primary_top15": dict(Counter(r["primary_category"] for r in rows).most_common(15)),
        "date_min": min(r["publication_date"] for r in rows),
        "date_max": max(r["publication_date"] for r in rows),
        "arxiv_package": getattr(arxiv, "__version__", "unknown"),
        "python": sys.version.split()[0],
        "finalized_at": datetime.now(timezone.utc).isoformat(),
    }
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"final corpus: {len(rows)} records -> {FINAL}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--finalize-only", action="store_true")
    args = p.parse_args()
    if not args.finalize_only:
        collect()
    finalize()


if __name__ == "__main__":
    main()
