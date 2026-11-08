#!/usr/bin/env python3
"""Step 3: evaluation for the camera-ready revision of paper 703.

Query-by-document retrieval on an arXiv corpus. Every eligible paper is a
query (title + abstract); the query paper itself is excluded from its ranking.
A retrieved paper is category-relevant when it has the query's primary arXiv
category. Graded nDCG gain is the Jaccard overlap of the category lists.

Query split (fixed before any scoring):
  dev   1,000 eligible queries drawn with seed 43  -> parameter sensitivity only
  test  all remaining eligible queries             -> every headline number

Outputs go to revision/results/<corpus>/.

Usage:
    python revision/scripts/03_evaluate.py --corpus may2026
    python revision/scripts/03_evaluate.py --corpus y2020_2024
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]           # .../article1_ictacs2026/revision
PAPER_DIR = ROOT.parents[1]                          # .../Paper
EMB = ROOT / "embeddings"
CORPORA = {
    # repository layout first, then the original project layout
    "may2026": (ROOT / "data" / "arxiv_may2026.jsonl") if (ROOT / "data" / "arxiv_may2026.jsonl").exists()
               else PAPER_DIR / "experiments" / "data" / "raw" / "arxiv_papers.jsonl",
    "y2020_2024": ROOT / "data" / "arxiv_2020_2024.jsonl",
}
# MiniLM title+abstract vectors of the May corpus used in the submitted paper.
ORIGINAL_MINILM = PAPER_DIR / "experiments" / "data" / "processed" / "embeddings.npy"
ORIGINAL_INDEX = PAPER_DIR / "experiments" / "data" / "processed" / "embedding_index.json"

MIN_CATEGORY_COUNT = 30
DEV_SIZE = 1000
DEV_SEED = 43
TOPN = 100            # list depth kept per query (MRR is computed to this depth)
CHUNK = 500
N_BOOT = 10_000
N_PERM = 10_000
STAT_SEED = 7

# Pre-specified defaults (identical to the submitted paper).
DEFAULTS = {"bm25_k1": 1.5, "bm25_b": 0.75, "tfidf_min_df": 3, "tfidf_max_df": 0.9,
            "tfidf_max_features": 20000, "lsa_dim": 128, "hybrid_w": 0.6}

DENSE_NAMES = {"minilm_ta": "MiniLM", "specter2_ta": "SPECTER2", "bge_small_ta": "BGE-small",
               "minilm_title": "MiniLM (title)", "minilm_abstract": "MiniLM (abstract)"}

TOKEN_RE = re.compile(r"[a-z0-9]+")


# ----------------------------------------------------------------------------- data
def load_corpus(name: str) -> list[dict]:
    with open(CORPORA[name], encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def field_text(p: dict, field: str) -> str:
    if field == "ta":
        return f"{p['title']} {p['abstract']}"
    if field == "title":
        return p["title"]
    if field == "abstract":
        return p["abstract"]
    raise ValueError(field)


def load_dense(corpus: str, papers: list[dict]) -> dict[str, np.ndarray]:
    ids = [p["arxiv_id"] for p in papers]
    out: dict[str, np.ndarray] = {}
    if corpus == "may2026" and ORIGINAL_MINILM.exists():
        idx = json.loads(ORIGINAL_INDEX.read_text(encoding="utf-8"))["arxiv_id_by_row"]
        assert [idx[str(i)] for i in range(len(ids))] == ids, "original MiniLM rows misaligned"
        out["minilm_ta"] = normalize(np.load(ORIGINAL_MINILM).astype(np.float32))
    ids_file = EMB / f"{corpus}__ids.txt"
    if ids_file.exists():
        stored = ids_file.read_text(encoding="utf-8").split()
        assert stored == ids, f"{ids_file} does not match the corpus order"
        for npy in sorted(EMB.glob(f"{corpus}__*.npy")):
            job = npy.stem.split("__", 1)[1]
            if job == "minilm_ta" and "minilm_ta" in out:
                # re-encoded copy: used only as a reproducibility check
                re_enc = normalize(np.load(npy).astype(np.float32))
                cos = np.sum(re_enc * out["minilm_ta"], axis=1)
                out["_minilm_reencode_cos"] = cos
                continue
            out[job] = normalize(np.load(npy).astype(np.float32))
    return out


# ----------------------------------------------------------------------------- scorers
def bm25_tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower())
            if t not in ENGLISH_STOP_WORDS and len(t) > 2 and not t.isdigit()]


class BM25:
    """Okapi BM25 as in the submitted paper; query terms counted once."""

    def __init__(self, texts: list[str], k1: float, b: float):
        cv = CountVectorizer(analyzer=bm25_tokenize, dtype=np.float32)
        tf = cv.fit_transform(texts).tocsr()               # docs x terms
        n = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(dl.mean())
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1.0 - b + b * dl[tf.row] / avgdl)
        w = idf[tf.col] * tf.data * (k1 + 1.0) / denom
        self.W = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.Q = sp.csr_matrix((np.ones_like(tf.data), (tf.row, tf.col)), shape=tf.shape)
        self.Q.data[:] = 1.0                                # unique query terms

    def scores(self, q: np.ndarray) -> np.ndarray:
        return (self.Q[q] @ self.W.T).toarray().astype(np.float32)


class Cosine:
    def __init__(self, X):
        self.X = X

    def scores(self, q: np.ndarray) -> np.ndarray:
        if sp.issparse(self.X):
            return (self.X[q] @ self.X.T).toarray().astype(np.float32)
        return self.X[q] @ self.X.T


def tfidf_matrix(texts: list[str], min_df=3, max_df=0.9, max_features=20000, sublinear=False):
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df, stop_words="english",
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b", sublinear_tf=sublinear,
                          dtype=np.float32)
    X = vec.fit_transform(texts)
    return X, len(vec.vocabulary_)


def lsa_matrix(X, dim: int):
    svd = TruncatedSVD(n_components=dim, random_state=42)
    Z = normalize(svd.fit_transform(X)).astype(np.float32)
    return Z, float(svd.explained_variance_ratio_.sum())


def _mask_self(S: np.ndarray, q: np.ndarray) -> np.ndarray:
    S = S.astype(np.float32, copy=True)
    S[np.arange(len(q)), q] = np.nan
    return S


def norm_minmax(S: np.ndarray) -> np.ndarray:
    lo = np.nanmin(S, axis=1, keepdims=True)
    hi = np.nanmax(S, axis=1, keepdims=True)
    return (S - lo) / np.maximum(hi - lo, 1e-12)


def norm_z(S: np.ndarray) -> np.ndarray:
    mu = np.nanmean(S, axis=1, keepdims=True)
    sd = np.nanstd(S, axis=1, keepdims=True)
    return (S - mu) / np.maximum(sd, 1e-12)


def rrf(S_a: np.ndarray, S_b: np.ndarray, k: int = 60) -> np.ndarray:
    def ranks(S):
        S = np.where(np.isnan(S), -np.inf, S)
        order = np.argsort(-S, axis=1, kind="stable")
        r = np.empty_like(order)
        r[np.arange(S.shape[0])[:, None], order] = np.arange(S.shape[1])[None, :]
        return r + 1
    return 1.0 / (k + ranks(S_a)) + 1.0 / (k + ranks(S_b))


class Hybrid:
    """w * norm(dense) + (1-w) * norm(BM25); normalisation excludes the query paper."""

    def __init__(self, dense: Cosine, bm25: BM25, w: float, mode: str = "minmax"):
        self.dense, self.bm25, self.w, self.mode = dense, bm25, w, mode

    def scores(self, q: np.ndarray) -> np.ndarray:
        D = _mask_self(self.dense.scores(q), q)
        B = _mask_self(self.bm25.scores(q), q)
        if self.mode == "rrf":
            return rrf(D, B)
        f = norm_minmax if self.mode == "minmax" else norm_z
        return self.w * f(D) + (1.0 - self.w) * f(B)


# ----------------------------------------------------------------------------- evaluation
class Judge:
    def __init__(self, papers: list[dict]):
        self.primary = np.array([p["primary_category"] for p in papers])
        cats = [frozenset(p.get("categories") or []) for p in papers]
        uniq = sorted(set(cats), key=lambda s: sorted(s))
        u_index = {c: i for i, c in enumerate(uniq)}
        self.set_id = np.array([u_index[c] for c in cats])
        U = len(uniq)
        jac = np.zeros((U, U), dtype=np.float32)
        for i, a in enumerate(uniq):
            for j, b in enumerate(uniq):
                un = len(a | b)
                jac[i, j] = len(a & b) / un if un else 0.0
        self.jac = jac
        self.set_count = np.bincount(self.set_id, minlength=U)
        self.disc = 1.0 / np.log2(np.arange(2, 12))
        self._ideal: dict[int, float] = {}

    def ideal_dcg(self, qi: int) -> float:
        s = self.set_id[qi]
        if s not in self._ideal:
            gains = np.repeat(self.jac[s], self.set_count)        # all docs incl. the query's own set
            gains = np.sort(gains)[::-1]
            # remove the query itself (it has gain 1 with its own set)
            gains = gains[1:11]
            self._ideal[s] = float(np.sum((2.0 ** gains - 1.0) * self.disc[: len(gains)]))
        return self._ideal[s]

    def metrics(self, qi: int, ranked: np.ndarray) -> dict:
        rel = self.primary[ranked] == self.primary[qi]
        g = self.jac[self.set_id[qi], self.set_id[ranked[:10]]]
        dcg = float(np.sum((2.0 ** g - 1.0) * self.disc))
        idcg = self.ideal_dcg(qi)
        bin_ideal = float(np.sum(self.disc[: min(10, int(np.sum(self.primary == self.primary[qi])) - 1)]))
        hit = np.flatnonzero(rel)
        return {
            "P@5": float(rel[:5].mean()),
            "P@10": float(rel[:10].mean()),
            "nDCG@10": dcg / idcg if idcg > 0 else float("nan"),
            "nDCG@10_bin": float(np.sum(rel[:10] * self.disc)) / bin_ideal if bin_ideal > 0 else float("nan"),
            "MRR": 1.0 / (hit[0] + 1) if hit.size else 0.0,
        }


def rank_all(scorer, queries: np.ndarray, depth: int = TOPN) -> np.ndarray:
    out = np.zeros((len(queries), depth), dtype=np.int32)
    for s in range(0, len(queries), CHUNK):
        q = queries[s:s + CHUNK]
        S = scorer.scores(q)
        S = np.where(np.isnan(S), -np.inf, S).astype(np.float32)
        S[np.arange(len(q)), q] = -np.inf
        part = np.argpartition(-S, depth, axis=1)[:, :depth]
        vals = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-vals, axis=1, kind="stable")
        out[s:s + len(q)] = np.take_along_axis(part, order, axis=1)
    return out


def evaluate(scorer, queries, judge) -> tuple[pd.DataFrame, np.ndarray]:
    ranked = rank_all(scorer, queries)
    df = pd.DataFrame([judge.metrics(int(qi), ranked[i]) for i, qi in enumerate(queries)])
    df.insert(0, "query_row", queries)
    return df, ranked


def mean_ci(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    m, sd = float(x.mean()), float(x.std(ddof=1))
    h = 1.96 * sd / math.sqrt(len(x))
    return {"mean": m, "ci_low": m - h, "ci_high": m + h, "n": int(len(x))}


def paired_stats(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    d = a - b
    d = d[~np.isnan(d)]
    n = len(d)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boot[i] = d[rng.integers(0, n, n)].mean()
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, n))
    null = (flips * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (N_PERM + 1)
    return {"diff": float(obs), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "p_perm": float(p), "n": int(n)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()
    t_start = time.perf_counter()
    out_dir = ROOT / "results" / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = load_corpus(args.corpus)
    N = len(papers)
    judge = Judge(papers)
    counts = Counter(judge.primary)
    eligible = np.array([i for i in range(N) if counts[judge.primary[i]] >= MIN_CATEGORY_COUNT])
    rng = np.random.default_rng(DEV_SEED)
    dev = np.sort(rng.choice(eligible, size=DEV_SIZE, replace=False))
    test = np.setdiff1d(eligible, dev)
    print(f"{args.corpus}: N={N}, eligible={len(eligible)}, dev={len(dev)}, test={len(test)}", flush=True)

    texts = {f: [field_text(p, f) for p in papers] for f in ("ta", "title", "abstract")}
    dense = load_dense(args.corpus, papers)
    reencode = dense.pop("_minilm_reencode_cos", None)

    # --- default sparse models (title + abstract) -----------------------------------------
    bm25 = BM25(texts["ta"], DEFAULTS["bm25_k1"], DEFAULTS["bm25_b"])
    X_tfidf, vocab = tfidf_matrix(texts["ta"])
    Z_lsa, lsa_var = lsa_matrix(X_tfidf, DEFAULTS["lsa_dim"])
    systems: dict[str, object] = {"BM25": bm25, "TF-IDF": Cosine(X_tfidf), "LSA-128": Cosine(Z_lsa)}
    for job in ("minilm_ta", "specter2_ta", "bge_small_ta"):
        if job in dense:
            systems[DENSE_NAMES[job]] = Cosine(dense[job])
    if "minilm_ta" in dense:
        systems["Hybrid MiniLM+BM25 (0.6)"] = Hybrid(Cosine(dense["minilm_ta"]), bm25, DEFAULTS["hybrid_w"])
    best_dense = None

    # --- main table on the test split ------------------------------------------------------
    per_query: dict[str, pd.DataFrame] = {}
    rankings: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    for name, sc in systems.items():
        t0 = time.perf_counter()
        df, rk = evaluate(sc, test, judge)
        per_query[name], rankings[name] = df, rk
        summary[name] = {m: mean_ci(df[m].to_numpy()) for m in ["P@5", "P@10", "nDCG@10", "nDCG@10_bin", "MRR"]}
        print(f"  test {name:28s} P@5={summary[name]['P@5']['mean']:.3f} nDCG@10={summary[name]['nDCG@10']['mean']:.3f} "
              f"MRR={summary[name]['MRR']['mean']:.3f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

    chance = float(np.mean([(counts[judge.primary[q]] - 1) / (N - 1) for q in test]))

    # --- paired statistics -----------------------------------------------------------------
    srng = np.random.default_rng(STAT_SEED)
    dense_names = [n for n in systems if n in DENSE_NAMES.values()]
    pairs = []
    for d in dense_names:
        for s in ("BM25", "TF-IDF", "LSA-128"):
            pairs.append((d, s))
    for i, a in enumerate(dense_names):
        for b in dense_names[i + 1:]:
            pairs.append((a, b))
    if "Hybrid MiniLM+BM25 (0.6)" in systems:
        pairs.append(("Hybrid MiniLM+BM25 (0.6)", "MiniLM"))
    pairs.append(("LSA-128", "TF-IDF"))
    pairs.append(("TF-IDF", "BM25"))
    stats: dict[str, dict] = {}
    for metric in ("P@5", "nDCG@10"):
        res = {f"{a} vs {b}": paired_stats(per_query[a][metric].to_numpy(), per_query[b][metric].to_numpy(), srng)
               for a, b in pairs}
        adj = holm({k: v["p_perm"] for k, v in res.items()})
        for k in res:
            res[k]["p_holm"] = adj[k]
        stats[metric] = res

    # --- per-category P@5 (test) -------------------------------------------------------------
    test_cat = judge.primary[test]
    top_cats = [c for c, _ in Counter(test_cat).most_common(4)]
    bycat = {}
    for c in top_cats:
        mask = test_cat == c
        bycat[c] = {"n": int(mask.sum()), "chance": (counts[c] - 1) / (N - 1),
                    **{name: float(per_query[name]["P@5"].to_numpy()[mask].mean()) for name in systems}}

    # --- list overlap ------------------------------------------------------------------------
    def jac10(a, b):
        return float(np.mean([len(set(x[:10]) & set(y[:10])) / len(set(x[:10]) | set(y[:10]))
                              for x, y in zip(rankings[a], rankings[b])]))
    overlap = {}
    names = list(systems)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a} | {b}"] = jac10(a, b)

    results = {"corpus": args.corpus, "N": N, "eligible": int(len(eligible)), "dev": int(len(dev)),
               "test": int(len(test)), "chance_P": chance, "tfidf_vocab": vocab, "lsa128_var": lsa_var,
               "defaults": DEFAULTS, "summary": summary, "stats": stats, "by_category": bycat,
               "top10_jaccard": overlap,
               "primary_share_corpus": {k: v / N for k, v in Counter(judge.primary).most_common(12)},
               "n_primary_categories": len(counts)}
    if reencode is not None:
        results["minilm_reencode_cosine"] = {"min": float(reencode.min()), "mean": float(reencode.mean())}

    # --- sensitivity & ablation on the dev split ---------------------------------------------
    if not args.skip_sensitivity:
        sens: dict[str, dict] = {}

        def dev_score(sc):
            df, _ = evaluate(sc, dev, judge)
            return {"P@5": float(df["P@5"].mean()), "nDCG@10": float(df["nDCG@10"].mean()),
                    "MRR": float(df["MRR"].mean())}

        sens["bm25"] = {f"k1={k1},b={b}": dev_score(BM25(texts["ta"], k1, b))
                        for k1 in (0.9, 1.2, 1.5, 2.0) for b in (0.4, 0.6, 0.75, 0.9)}
        print("  sensitivity: bm25 done", flush=True)
        sens["tfidf"] = {}
        for min_df in (1, 3, 5):
            for sub in (False, True):
                X, _ = tfidf_matrix(texts["ta"], min_df=min_df, sublinear=sub)
                sens["tfidf"][f"min_df={min_df},sublinear={sub}"] = dev_score(Cosine(X))
        sens["tfidf"]["min_df=3,max_features=None"] = dev_score(Cosine(tfidf_matrix(texts["ta"], max_features=None)[0]))
        print("  sensitivity: tfidf done", flush=True)
        sens["lsa"] = {}
        for dim in (64, 128, 256, 512):
            Z, var = lsa_matrix(X_tfidf, dim)
            sens["lsa"][f"dim={dim}"] = {**dev_score(Cosine(Z)), "explained_var": var}
        print("  sensitivity: lsa done", flush=True)
        # hybrid weights / normalisation, for every available dense model
        sens["hybrid"] = {}
        for job in ("minilm_ta", "specter2_ta", "bge_small_ta"):
            if job not in dense:
                continue
            dn = Cosine(dense[job])
            for mode in ("minmax", "z"):
                for w in np.round(np.arange(0.0, 1.01, 0.1), 1):
                    sens["hybrid"][f"{DENSE_NAMES[job]}|{mode}|w={w}"] = dev_score(Hybrid(dn, bm25, float(w), mode))
            sens["hybrid"][f"{DENSE_NAMES[job]}|rrf"] = dev_score(Hybrid(dn, bm25, 0.5, "rrf"))
        print("  sensitivity: hybrid done", flush=True)
        # field ablation
        abl = {}
        for f in ("title", "abstract", "ta"):
            abl[f"BM25|{f}"] = dev_score(BM25(texts[f], DEFAULTS["bm25_k1"], DEFAULTS["bm25_b"]))
            Xf, _ = tfidf_matrix(texts[f], min_df=1 if f == "title" else 3)
            abl[f"TF-IDF|{f}"] = dev_score(Cosine(Xf))
            abl[f"LSA-128|{f}"] = dev_score(Cosine(lsa_matrix(Xf, 128)[0]))
        for job, f in (("minilm_title", "title"), ("minilm_abstract", "abstract"), ("minilm_ta", "ta")):
            if job in dense:
                abl[f"MiniLM|{f}"] = dev_score(Cosine(dense[job]))
        sens["ablation_fields"] = abl
        results["dev_sensitivity"] = sens

        # configurations selected on dev by nDCG@10, then scored once on test
        def best(group, prefix=""):
            cand = {k: v for k, v in sens[group].items() if k.startswith(prefix)}
            return max(cand, key=lambda k: cand[k]["nDCG@10"])
        tuned = {}
        k_bm = best("bm25")
        k1, b = (float(x.split("=")[1]) for x in k_bm.split(","))
        bm_t = BM25(texts["ta"], k1, b)
        tuned[f"BM25 ({k_bm})"] = bm_t
        k_tf = best("tfidf")
        if "max_features=None" in k_tf:
            X_t = tfidf_matrix(texts["ta"], max_features=None)[0]
        else:
            md, sub = k_tf.split(",")
            X_t = tfidf_matrix(texts["ta"], min_df=int(md.split("=")[1]), sublinear=sub.endswith("True"))[0]
        tuned[f"TF-IDF ({k_tf})"] = Cosine(X_t)
        k_lsa = best("lsa")
        tuned[f"LSA ({k_lsa})"] = Cosine(lsa_matrix(X_tfidf, int(k_lsa.split("=")[1]))[0])
        for job in ("minilm_ta", "specter2_ta", "bge_small_ta"):
            if job in dense:
                k_h = best("hybrid", DENSE_NAMES[job] + "|")
                _, mode, wpart = (k_h.split("|") + ["w=0.5"])[:3]
                tuned[f"Hybrid {k_h}"] = Hybrid(Cosine(dense[job]), bm25, float(wpart.split("=")[1]), mode)
        results["test_tuned"] = {}
        for name, sc in tuned.items():
            df, _ = evaluate(sc, test, judge)
            results["test_tuned"][name] = {m: mean_ci(df[m].to_numpy()) for m in ["P@5", "nDCG@10", "MRR"]}
            per_query[f"tuned: {name}"] = df
            print(f"  tuned {name:40s} P@5={df['P@5'].mean():.3f} nDCG@10={df['nDCG@10'].mean():.3f}", flush=True)

        # hybrid weight curve on the TEST split too (reported, not used for selection)
        if "minilm_ta" in dense:
            dn = Cosine(dense["minilm_ta"])
            for mode in ("minmax", "z"):
                curve = {}
                for w in np.round(np.arange(0.0, 1.01, 0.1), 1):
                    df, _ = evaluate(Hybrid(dn, bm25, float(w), mode), test, judge)
                    curve[str(w)] = {m: mean_ci(df[m].to_numpy()) for m in ("P@5", "nDCG@10")}
                results[f"test_hybrid_curve_minilm_{mode}"] = curve
            df, _ = evaluate(Hybrid(dn, bm25, 0.5, "rrf"), test, judge)
            results["test_hybrid_rrf_minilm"] = {m: mean_ci(df[m].to_numpy()) for m in ("P@5", "nDCG@10")}
            print("  test hybrid curves done", flush=True)

    # --- save -------------------------------------------------------------------------------
    long = pd.concat([d.assign(method=n) for n, d in per_query.items()])
    long["arxiv_id"] = [papers[i]["arxiv_id"] for i in long["query_row"]]
    long.to_csv(out_dir / "per_query_test.csv", index=False)
    np.savez_compressed(out_dir / "rankings_test_top100.npz", queries=test,
                        **{n.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p"): r
                           for n, r in rankings.items()})
    (out_dir / "dev_query_rows.txt").write_text("\n".join(papers[i]["arxiv_id"] for i in dev) + "\n", encoding="utf-8")
    results["runtime_s"] = round(time.perf_counter() - t_start, 1)
    results["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                           "scikit-learn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__}
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"wrote {out_dir} in {results['runtime_s']} s")


if __name__ == "__main__":
    main()
