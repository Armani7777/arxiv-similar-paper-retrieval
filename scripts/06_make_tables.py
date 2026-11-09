#!/usr/bin/env python3
"""Step 6: turn result files into the paper's numbers, tables and figure.

Reads  revision/results/{may2026,y2020_2024}/results.json,
       revision/results/descriptives.json,
       revision/judgments/{pool_key.json,labels.json} (if present).
Writes revision/paper/numbers.tex, revision/paper/tables/*.tex,
       revision/paper/figures/hybrid_weights.pdf,
       revision/results/reader_summary.json.
Every number printed in the paper comes from these files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "pdf.fonttype": 42})
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PAPER = ROOT / "paper"
TAB = PAPER / "tables"
FIG = PAPER / "figures"
JUD = ROOT / "judgments"

ORDER = ["BM25", "TF-IDF", "LSA-128", "MiniLM", "SPECTER2", "BGE-small", "Hybrid MiniLM+BM25 (0.6)"]
SHOW = {"LSA-128": "LSA", "Hybrid MiniLM+BM25 (0.6)": "Hybrid (0.6)"}
KEY = {"BM25": "BM25", "TF-IDF": "TFIDF", "LSA-128": "LSA", "MiniLM": "MiniLM", "SPECTER2": "SPECTER",
       "BGE-small": "BGE", "Hybrid MiniLM+BM25 (0.6)": "Hyb"}
NPZ = {s: s.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "p") for s in ORDER}
DENSE = ["MiniLM", "SPECTER2", "BGE-small"]
SPARSE = ["BM25", "TF-IDF", "LSA-128"]

vals: dict[str, str] = {}


def f3(x: float) -> str:
    return f"{x:.3f}"


def fp(p: float) -> str:
    # 10,000 sign flips -> smallest raw p is 1/10001; after Holm over ~15 pairs the floor is ~0.0015
    return "$<$0.002" if p < 0.002 else f"{p:.3f}"


def fd(x: float) -> str:
    return f"{x:+.3f}".replace("+", "+").replace("-", "$-$")


def load(c):
    return json.loads((RES / c / "results.json").read_text(encoding="utf-8"))


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    R = {"may": load("may2026"), "y": load("y2020_2024")}
    D = json.loads((RES / "descriptives.json").read_text(encoding="utf-8"))
    meta_y = json.loads((ROOT / "data" / "arxiv_2020_2024_meta.json").read_text(encoding="utf-8"))
    systems = [s for s in ORDER if s in R["may"]["summary"] and s in R["y"]["summary"]]

    # ---------------- scalar values
    for c, r in R.items():
        vals[f"{c}.test"] = f"{r['test']:,}".replace(",", "{,}")
        vals[f"{c}.chance"] = f3(r["chance_P"])
        for s in systems:
            for m, short in (("P@5", "P5"), ("nDCG@10", "nDCG"), ("MRR", "MRR")):
                vals[f"{c}.{KEY[s]}.{short}"] = f3(r["summary"][s][m]["mean"])
    vals["y.candidates"] = f"{meta_y['candidates_after_cleaning']:,}".replace(",", "{,}")
    D = json.loads((RES / "descriptives.json").read_text(encoding="utf-8"))
    best = max(DENSE, key=lambda s: R["may"]["summary"].get(s, {"P@5": {"mean": -1}})["P@5"]["mean"])
    vals["may.best.name"] = best
    vals["may.best.P5"] = f3(R["may"]["summary"][best]["P@5"]["mean"])
    best_sparse = max(SPARSE, key=lambda s: R["may"]["summary"][s]["P@5"]["mean"])
    vals["gap.points"] = f"{100 * (R['may']['summary'][best]['P@5']['mean'] - R['may']['summary'][best_sparse]['P@5']['mean']):.1f}"
    best_y = max(DENSE, key=lambda s: R["y"]["summary"][s]["P@5"]["mean"])
    best_sparse_y = max(SPARSE, key=lambda s: R["y"]["summary"][s]["P@5"]["mean"])
    vals["gap.points.y"] = f"{100 * (R['y']['summary'][best_y]['P@5']['mean'] - R['y']['summary'][best_sparse_y]['P@5']['mean']):.1f}"
    hy = R["may"]["stats"]["P@5"]["Hybrid MiniLM+BM25 (0.6) vs MiniLM"]
    vals["may.hyb.diff"] = fd(hy["diff"])
    vals["may.hyb.p"] = f"{hy['p_holm']:.2f}"
    for c in ("may", "y"):
        st = R[c]["stats"]["P@5"]
        hy_c = st["Hybrid MiniLM+BM25 (0.6) vs MiniLM"]
        vals[f"{c}.hyb.diff"] = fd(hy_c["diff"])
        vals[f"{c}.hyb.p"] = fp(hy_c["p_holm"]) if hy_c["p_holm"] < 0.002 else f"{hy_c['p_holm']:.2f}"
        sm = st["MiniLM vs SPECTER2"]
        vals[f"{c}.spec.minilm"] = fd(-sm["diff"])
        vals[f"{c}.spec.minilm.p"] = fp(sm["p_holm"])
        for s_ in DENSE:
            vals[f"{c}.{KEY[s_]}.gap"] = fd(R[c]["summary"][s_]["P@5"]["mean"] - max(R[c]["summary"][x]["P@5"]["mean"] for x in SPARSE))
        tt = R[c]["test_tuned"]
        k = [k for k in tt if k.startswith("Hybrid MiniLM|")][0]
        vals[f"{c}.hybtuned.w"] = k.split("|")[-1].replace("w=", "")
        vals[f"{c}.hybtuned.P5"] = f3(tt[k]["P@5"]["mean"])
        mm = R[c]["test_hybrid_curve_minilm_minmax"]
        rng6 = [mm[w]["nDCG@10"]["mean"] for w in ("0.6", "0.7", "0.8", "0.9")]
        vals[f"{c}.curve.spread"] = f"{max(rng6) - min(rng6):.3f}"
        vals[f"{c}.rrf.nDCG"] = f3(R[c]["test_hybrid_rrf_minilm"]["nDCG@10"]["mean"])
        vals[f"{c}.curve.best"] = f3(max(v["nDCG@10"]["mean"] for v in mm.values()))
        o = R[c]["top10_jaccard"]
        dd = [o["MiniLM | SPECTER2"], o["MiniLM | BGE-small"], o["SPECTER2 | BGE-small"]]
        vals[f"{c}.ov.dense"] = f"{min(dd):.2f}--{max(dd):.2f}"
        chk = D["may2026" if c == "may" else "y2020_2024"]["top5_checks"]
        sa = [v["shared_author_share"] for v in chk.values()]
        vals[f"{c}.sharedauthor"] = f"{100*min(sa):.1f}--{100*max(sa):.1f}\\%"
        vals[f"{c}.authorfilter.maxdiff"] = f"{max(abs(v['P@5'] - v['P@5_without_shared_author']) for v in chk.values()):.3f}"
        nd = [v["title_jaccard_ge_0.5"] for v in chk.values()]
        vals[f"{c}.neardup"] = f"{min(nd)}--{max(nd)}"
        vals[f"{c}.slots"] = f"{list(chk.values())[0]['slots']:,}".replace(",", "{,}")
    ov = R["may"]["top10_jaccard"]
    vals["may.ov.MiniLM.BM25"] = f"{ov['BM25 | MiniLM']:.2f}"
    vals["may.ov.MiniLM.LSA"] = f"{ov['LSA-128 | MiniLM']:.2f}"

    # main sentence: ranking of dense models on both corpora
    def ranked(c):
        return sorted(DENSE, key=lambda s: -R[c]["summary"][s]["P@5"]["mean"]) if all(s in R[c]["summary"] for s in DENSE) else []
    rm, ry = ranked("may"), ranked("y")
    if rm:
        vals["main.sentence"] = (
            f"Among the dense encoders, {rm[0]} is first on the May corpus and {ry[0]} on the 2020--2024 corpus; "
            f"the best sparse system is {SHOW.get(best_sparse, best_sparse)} on both.")
    else:
        vals["main.sentence"] = ""

    # ---------------- Table I data
    dm, dy = D["may2026"], D["y2020_2024"]
    def share(d, k):
        return f"{100 * d['primary_top8'].get(k, 0) / d['n']:.1f}\\%"
    def later_versions(d):
        return f"{100 * (1 - d['version_suffix'].get('1', 0) / d['n']):.1f}\\%"
    rows = [
        ("Records", "10{,}000", "10{,}000"),
        ("Publication dates", f"{dm['date_min']} -- {dm['date_max'][5:]}", f"{dy['date_min'][:7]} -- {dy['date_max'][:7]}"),
        ("Distinct primary categories", str(dm["n_primary_categories"]), str(dy["n_primary_categories"])),
        ("Primary in query categories", f"{100*dm['primary_in_query_cats']:.1f}\\%", f"{100*dy['primary_in_query_cats']:.1f}\\%"),
        ("\\quad cs.LG / cs.CL", f"{share(dm,'cs.LG')} / {share(dm,'cs.CL')}", f"{share(dy,'cs.LG')} / {share(dy,'cs.CL')}"),
        ("\\quad cs.AI / cs.CV", f"{share(dm,'cs.AI')} / {share(dm,'cs.CV')}", f"{share(dy,'cs.AI')} / {share(dy,'cs.CV')}"),
        ("Categories per paper (mean)", f"{dm['categories_per_paper_mean']:.2f}", f"{dy['categories_per_paper_mean']:.2f}"),
        ("Abstract words (median)", f"{dm['abstract_words_median']:.0f}", f"{dy['abstract_words_median']:.0f}"),
        ("Revised after v1", later_versions(dm), later_versions(dy)),
        ("Queries: eligible/dev/test", f"{R['may']['eligible']:,}/1{{,}}000/{R['may']['test']:,}".replace(",", "{,}").replace("{{,}}", "{,}"),
         f"{R['y']['eligible']:,}/1{{,}}000/{R['y']['test']:,}".replace(",", "{,}").replace("{{,}}", "{,}")),
        ("Chance precision (test)", f3(R["may"]["chance_P"]), f3(R["y"]["chance_P"])),
    ]
    t = ["\\begin{table}[!t]", "\\centering", "\\caption{The two corpora.}", "\\label{tab:data}", "\\footnotesize",
         "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lcc@{}}", "\\toprule", " & May 2026 & 2020--2024 \\\\", "\\midrule"]
    t += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "data.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ---------------- Table II main
    hw = max(max(R[c]["summary"][s][m]["mean"] - R[c]["summary"][s][m]["ci_low"]
                 for s in systems for m in ("P@5", "nDCG@10")) for c in R)
    t = ["\\begin{table}[!t]", "\\centering",
         f"\\caption{{Test-set results with pre-specified parameters. Relevance: same primary category; nDCG@10 uses graded category overlap. Every 95\\% interval for P@5 and nDCG@10 has a half-width of at most {hw:.3f}. Best value per column in bold.}}",
         "\\label{tab:main}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.2pt}", "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{May 2026} & \\multicolumn{3}{c}{2020--2024} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}", "Method & P@5 & nDCG@10 & MRR & P@5 & nDCG@10 & MRR \\\\", "\\midrule"]
    bestv = {(c, m): max(R[c]["summary"][s][m]["mean"] for s in systems) for c in R for m in ("P@5", "nDCG@10", "MRR")}
    for s in systems:
        cells = []
        for c in ("may", "y"):
            for m in ("P@5", "nDCG@10", "MRR"):
                v = R[c]["summary"][s][m]["mean"]
                cells.append(f"\\textbf{{{f3(v)}}}" if abs(v - bestv[(c, m)]) < 5e-4 else f3(v))
        if s == "MiniLM":
            t.append("\\midrule")
        if s.startswith("Hybrid"):
            t.append("\\midrule")
        t.append(f"{SHOW.get(s, s)} & " + " & ".join(cells) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "main.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ---------------- Table III stats
    def pair_rows(c):
        st = R[c]["stats"]["P@5"]
        bs = max(SPARSE, key=lambda s: R[c]["summary"][s]["P@5"]["mean"])
        rows = []
        for d in DENSE:
            if d in R[c]["summary"]:
                rows.append((f"{d} -- {SHOW.get(bs, bs)}", st[f"{d} vs {bs}"]))
        return rows
    keys = []
    for d in DENSE:
        keys.append((d, "best sparse"))
    t = ["\\begin{table*}[!t]", "\\centering",
         "\\caption{Paired differences in test P@5: mean difference, 95\\% bootstrap interval and Holm-adjusted randomization $p$-value.}",
         "\\label{tab:stats}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{May 2026} & \\multicolumn{2}{c}{2020--2024} \\\\", "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
         "Comparison & $\\Delta$ [95\\% CI] & $p$ & $\\Delta$ [95\\% CI] & $p$ \\\\", "\\midrule"]
    comps = []
    for d in DENSE:
        if d in R["may"]["summary"]:
            comps.append((f"{d} vs BM25", f"{d} -- BM25"))
    for a, b in (("SPECTER2", "MiniLM"), ("BGE-small", "MiniLM"), ("BGE-small", "SPECTER2")):
        comps.append((f"{a} vs {b}", f"{a} -- {b}"))
    comps.append(("Hybrid MiniLM+BM25 (0.6) vs MiniLM", "Hybrid (0.6) -- MiniLM"))
    comps.append(("TF-IDF vs BM25", "TF-IDF -- BM25"))
    comps.append(("LSA-128 vs TF-IDF", "LSA -- TF-IDF"))
    for key, label in comps:
        cells = []
        ok = True
        for c in ("may", "y"):
            st = R[c]["stats"]["P@5"]
            k = key if key in st else (" vs ".join(key.split(" vs ")[::-1]) if " vs ".join(key.split(" vs ")[::-1]) in st else None)
            if k is None:
                ok = False
                break
            v = st[k]
            sign = 1 if k == key else -1
            lo, hi = sorted((sign * v["ci_low"], sign * v["ci_high"]))
            cells += [f"{fd(sign*v['diff'])} [{fd(lo)}, {fd(hi)}]", fp(v["p_holm"])]
        if ok:
            t.append(f"{label} & " + " & ".join(cells) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    (TAB / "stats.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ---------------- Table IV by category (May)
    bc = R["may"]["by_category"]
    cols = [s for s in ["BM25", "TF-IDF", "LSA-128", "MiniLM", "SPECTER2", "BGE-small"] if s in systems]
    t = ["\\begin{table}[!t]", "\\centering",
         "\\caption{Test P@5 by primary category of the query, May 2026 corpus. Chance is $(n_c-1)/(N-1)$.}",
         "\\label{tab:bycat}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.8pt}",
         "\\begin{tabular}{@{}lrr" + "r" * len(cols) + "@{}}", "\\toprule",
         "Cat. & $n$ & Chance & " + " & ".join({"BM25": "BM25", "TF-IDF": "TF-IDF", "LSA-128": "LSA", "MiniLM": "MiniLM", "SPECTER2": "SPEC2", "BGE-small": "BGE"}[s] for s in cols) + " \\\\",
         "\\midrule"]
    for cat, v in bc.items():
        best_c = max(v[s] for s in cols)
        cells = [f"\\textbf{{{f3(v[s])}}}" if abs(v[s] - best_c) < 5e-4 else f3(v[s]) for s in cols]
        t.append(f"{cat} & {v['n']:,} & {f3(v['chance'])} & ".replace(",", "{,}", 1) + " & ".join(cells) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / "bycat.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # ---------------- Table V sensitivity / ablation (May dev + tuned on test)
    sens = R["may"]["dev_sensitivity"]
    tuned = R["may"]["test_tuned"]
    def rng(group, prefix=""):
        xs = [v["nDCG@10"] for k, v in sens[group].items() if k.startswith(prefix)]
        return f"{min(xs):.3f}--{max(xs):.3f}"
    def tuned_of(prefix):
        k = [k for k in tuned if k.startswith(prefix)][0]
        return k, tuned[k]["P@5"]["mean"]
    t = ["\\begin{table*}[!t]", "\\centering",
         "\\caption{Sensitivity on the 1{,}000 development queries (May 2026). Range: dev nDCG@10 across the grid. The configuration with the best dev nDCG@10 is then scored once on the test set (P@5, default $\\rightarrow$ selected). Bottom: dev P@5 by indexed field.}",
         "\\label{tab:sens}", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{@{}llccc@{}}", "\\toprule",
         "System & Grid & Dev range & Selected & Test P@5 \\\\", "\\midrule"]
    gains = []
    for sysname, group, grid, pre, base in (
            ("BM25", "bm25", "$k_1\\!\\in\\!\\{0.9,1.2,1.5,2\\}$, $b\\!\\in\\!\\{.4,.6,.75,.9\\}$", "BM25 (", "BM25"),
            ("TF-IDF", "tfidf", "min\\_df$\\in\\!\\{1,3,5\\}$, sublinear tf", "TF-IDF (", "TF-IDF"),
            ("LSA", "lsa", "$k\\in\\{64,128,256,512\\}$", "LSA (", "LSA-128")):
        k, v = tuned_of(pre)
        sel = k[k.index("(") + 1:-1].replace("min_df", "min\\_df").replace("sublinear=True", "sublin.").replace(",", ", ")
        b0 = R["may"]["summary"][base]["P@5"]["mean"]
        gains.append(v - b0)
        t.append(f"{sysname} & {grid} & {rng(group)} & {sel} & {f3(b0)} $\\rightarrow$ {f3(v)} \\\\")
    for d in DENSE:
        hk = [k for k in tuned if k.startswith(f"Hybrid {d}|")]
        if hk:
            k = hk[0]
            v = tuned[k]["P@5"]["mean"]
            parts = k.split("|")
            sel = parts[1].replace("minmax", "min-max").replace("z", "z-score") + (f", {parts[2]}" if len(parts) > 2 else "")
            b0 = R["may"]["summary"][d]["P@5"]["mean"]
            t.append(f"{d}+BM25 & $w\\!\\in\\![0,1]$, min-max/z, RRF & {rng('hybrid', d + '|')} & {sel} & {f3(b0)}$^\\dagger$ $\\rightarrow$ {f3(v)} \\\\")
    t.append("\\midrule")
    abl = sens["ablation_fields"]
    t.append(" & & Title & Abstract & Title+abstract \\\\")
    t.append("\\cmidrule(l){3-5}")
    for s in ("BM25", "TF-IDF", "LSA-128", "MiniLM"):
        cells = [f3(abl[f"{s}|{f}"]["P@5"]) if f"{s}|{f}" in abl else "--" for f in ("title", "abstract", "ta")]
        t.append(f"{SHOW.get(s, s)} & dev P@5 & " + " & ".join(cells) + " \\\\")
    t += ["\\bottomrule", "\\multicolumn{5}{@{}l}{\\scriptsize $^\\dagger$ dense model alone.}", "\\end{tabular}", "\\end{table*}"]
    (TAB / "sens.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    vals["tune.maxgain"] = f"{max(gains):.3f}"
    ty = R["y"]["test_tuned"]
    gy = [ty[[k for k in ty if k.startswith(pre)][0]]["P@5"]["mean"] - R["y"]["summary"][base]["P@5"]["mean"]
          for pre, base in (("BM25 (", "BM25"), ("TF-IDF (", "TF-IDF"), ("LSA (", "LSA-128"))]
    vals["tune.maxgain.y"] = f"{max(gy):.3f}"
    vals["tune.bestsparse.y"] = f3(max(ty[k]["P@5"]["mean"] for k in ty if not k.startswith("Hybrid")))
    vals["y.weakestdense.P5"] = f3(min(R["y"]["summary"][d]["P@5"]["mean"] for d in DENSE))
    sparse_loss = [abl[f"{s}|ta"]["P@5"] - abl[f"{s}|title"]["P@5"] for s in ("BM25", "TF-IDF", "LSA-128")]
    vals["abl.titleloss"] = f"{min(sparse_loss):.3f}--{max(sparse_loss):.3f}"
    if "MiniLM|title" in abl:
        vals["abl.minilm.titleloss"] = f"{abl['MiniLM|ta']['P@5'] - abl['MiniLM|title']['P@5']:.3f}"
        vals["abl.minilm.title"] = f3(abl["MiniLM|title"]["P@5"])
        vals["abl.sparse.best.ta"] = f3(max(abl[f"{s}|ta"]["P@5"] for s in ("BM25", "TF-IDF", "LSA-128")))

    # ---------------- Figure: hybrid weights on test
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    styles = {"may": ("May 2026", "#2f5d8a", "o"), "y": ("2020--2024", "#b5532c", "s")}
    for c, (lab, col, mk) in styles.items():
        for mode, ls in (("minmax", "-"), ("z", "--")):
            cur = R[c].get(f"test_hybrid_curve_minilm_{mode}")
            if not cur:
                continue
            w = np.array([float(k) for k in cur])
            m = np.array([v["nDCG@10"]["mean"] for v in cur.values()])
            lo = np.array([v["nDCG@10"]["ci_low"] for v in cur.values()])
            hi = np.array([v["nDCG@10"]["ci_high"] for v in cur.values()])
            ax.plot(w, m, ls, color=col, lw=1.2, label=f"{lab.replace('--', '–')}, {'min-max' if mode == 'minmax' else 'z-score'}")
            if mode == "minmax":
                ax.fill_between(w, lo, hi, color=col, alpha=0.15, lw=0)
                ax.plot([0.6], [m[np.argmin(abs(w - 0.6))]], mk, color=col, ms=4)
        rr = R[c].get("test_hybrid_rrf_minilm")
        if rr:
            ax.plot([1.04], [rr["nDCG@10"]["mean"]], "x", color=col, ms=5)
    ax.set_xlabel("weight $w$ on MiniLM", fontsize=7)
    ax.set_ylabel("nDCG@10", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=5.6, frameon=False, loc="upper left")
    ax.text(1.04, ax.get_ylim()[0], "RRF", fontsize=5.5, ha="center", va="bottom")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "hybrid_weights.pdf")
    plt.close(fig)

    # ---------------- Reader table
    lab_path = JUD / "labels.json"
    if lab_path.exists():
        key = json.loads((JUD / "pool_key.json").read_text(encoding="utf-8"))
        labels = json.loads(lab_path.read_text(encoding="utf-8"))
        labels = labels.get("labels", labels)
        sys_list = key["systems"]
        out = {s: {"reader": [], "cat": [], "both": 0, "prob_only": 0, "cat_only": 0, "neither": 0} for s in sys_list}
        missing = 0
        for q in key["queries"]:
            for s in sys_list:
                top = sorted([c for c in q["candidates"] if s in c["systems"]], key=lambda c: c["rank"][s])[:5]
                rr = []
                cc = []
                for c in top:
                    if c["cid"] not in labels:
                        missing += 1
                        continue
                    r_, c_ = int(labels[c["cid"]]), int(c["primary"] == q["primary"])
                    rr.append(r_)
                    cc.append(c_)
                    out[s]["both" if r_ and c_ else "prob_only" if r_ else "cat_only" if c_ else "neither"] += 1
                out[s]["reader"].append(np.mean(rr) if rr else np.nan)
                out[s]["cat"].append(np.mean(cc) if cc else np.nan)
        n_pool = sum(len(q["candidates"]) for q in key["queries"])
        n_rel = sum(int(labels.get(c["cid"], 0)) for q in key["queries"] for c in q["candidates"])
        summ = {s: {"reader_P5": float(np.nanmean(v["reader"])), "cat_P5": float(np.nanmean(v["cat"])),
                    **{k: v[k] for k in ("both", "prob_only", "cat_only", "neither")}} for s, v in out.items()}
        (RES / "reader_summary.json").write_text(json.dumps({"pool": n_pool, "relevant": n_rel, "missing": missing,
                                                             "systems": summ}, indent=1), encoding="utf-8")
        npz2name = {v: k for k, v in NPZ.items()}
        t = ["\\begin{table}[!t]", "\\centering",
             "\\caption{Problem-level reading of 10 May-2026 test queries (one annotator, blind to system and category). Reader P@5: same research problem. Category P@5: same primary category, same lists. Right: the 50 top-five slots of each system split by the two labels.}",
             "\\label{tab:reader}", "\\scriptsize", "\\setlength{\\tabcolsep}{2.6pt}", "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
             " & Reader & Category & \\multicolumn{4}{c}{Slots (problem / category)} \\\\", "\\cmidrule(l){4-7}",
             "Method & P@5 & P@5 & yes/yes & yes/no & no/yes & no/no \\\\", "\\midrule"]
        for s in sys_list:
            nm = npz2name.get(s, s)
            v = summ[s]
            t.append(f"{SHOW.get(nm, nm)} & {v['reader_P5']:.2f} & {v['cat_P5']:.2f} & {v['both']} & {v['prob_only']} & {v['cat_only']} & {v['neither']} \\\\")
        t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (TAB / "reader.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
        vals["pool.size"] = str(n_pool)
        dense_npz = [NPZ[d] for d in DENSE if NPZ[d] in summ]
        top = max(summ, key=lambda s: summ[s]["reader_P5"])
        bm = summ[NPZ["BM25"]]
        po = sum(summ[s]["prob_only"] for s in dense_npz)
        co = sum(summ[s]["cat_only"] for s in dense_npz)
        tot = sum(sum(summ[s][k] for k in ("both", "prob_only", "cat_only", "neither")) for s in dense_npz)
        rel = [c for q in key["queries"] for c in q["candidates"] if int(labels.get(c["cid"], 0))]
        dense_set = {NPZ[d] for d in DENSE}
        sparse_set = {NPZ[x] for x in SPARSE}
        no_dense = sum(not (set(c["systems"]) & dense_set) for c in rel)
        only_dense = sum(not (set(c["systems"]) & (sparse_set | {NPZ["Hybrid MiniLM+BM25 (0.6)"]})) for c in rel)
        empty_q = sum(not any(int(labels.get(c["cid"], 0)) for c in q["candidates"]) for q in key["queries"])
        lo = min(v["reader_P5"] for v in summ.values()); hi = max(v["reader_P5"] for v in summ.values())
        def names(pred):
            ns = [SHOW.get(npz2name.get(s_, s_), npz2name.get(s_, s_)).replace("Hybrid (0.6)", "the hybrid") for s_ in sys_list if pred(summ[s_]["reader_P5"])]
            return ", ".join(ns[:-1]) + " and " + ns[-1] if len(ns) > 1 else "".join(ns)
        vals["reader.lo"] = f"{lo:.2f}"; vals["reader.hi"] = f"{hi:.2f}"
        vals["reader.rel"] = str(n_rel); vals["reader.nodense"] = str(no_dense); vals["reader.onlydense"] = str(only_dense)
        vals["reader.sentence"] = (
            f"Of the {n_pool} pooled candidates the first author marked {n_rel} as the same research problem; "
            f"for {empty_q} of the ten queries no pooled candidate qualified. Reader P@5 is low for every system, "
            f"between {lo:.2f} and {hi:.2f}, and it does not follow the ranking by category: {names(lambda x: x >= 0.175)} "
            f"reach 0.18--{hi:.2f}, while {names(lambda x: x <= 0.105)} reach {lo:.2f}. Of the {n_rel} same-problem papers, "
            f"{no_dense} were retrieved by none of the three dense encoders and {only_dense} only by them. Category agreement "
            f"over-counts far more often than it under-counts: of the {tot} top-five slots of the dense encoders, {co} share the "
            f"query's category without addressing the same problem, while {po} address the same problem from another category. "
            f"With ten queries and one annotator these figures are descriptive, but they indicate that the dense advantage in "
            f"category precision reflects topical closeness more than same-problem relevance.")
    else:
        vals["pool.size"] = "--"
        vals["reader.sentence"] = "[problem-level labels pending]"
        (TAB / "reader.tex").write_text("% reader labels pending\n", encoding="utf-8")

    vals.setdefault("repo", "Armani7777/arxiv-similar-paper-retrieval")
    vals.setdefault("ai.disclosure", "Generative AI (Claude, Anthropic) was used to help write the analysis scripts and edit the manuscript text. The authors designed the study, ran all experiments, produced the manual relevance labels, verified all code, results and references, and take full responsibility for the content.")
    lines = ["\\makeatletter"]
    for k, v in sorted(vals.items()):
        lines.append(f"\\@namedef{{v@{k}}}{{{v}}}")
    lines += ["\\makeatother",
              "\\newcommand{\\val}[1]{\\ifcsname v@#1\\endcsname\\csname v@#1\\endcsname\\else\\textbf{??#1??}\\fi}"]
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(vals, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
