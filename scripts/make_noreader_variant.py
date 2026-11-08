#!/usr/bin/env python3
"""Variant of the paper without the problem-level reading (used only if no valid labels exist)."""
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
s = src.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


rep(r"""A blind problem-level reading of 10 queries shows that category agreement and same-problem relevance often diverge, so category precision should be read as a proxy rather than as a measure of usefulness.""",
    r"""Category agreement is only a proxy for same-problem relevance, so the absolute precision values should be read as a comparison between systems rather than as the share of useful results.""")
rep(r"""\item a blind, problem-level reading of pooled results that measures how often the category proxy and same-problem relevance disagree.
""", "")
rep(r"""Code, query lists, per-query scores and the manual labels are publicly released""", r"""Code, query lists, per-query scores and rankings are publicly released""")
rep(r"""\node[box=24mm, anchor=north east] (cat) at ([yshift=-5mm, xshift=-1.5mm]de.south) {category proxy:\\all test queries};
\node[box=24mm, anchor=north west] (man) at ([yshift=-5mm, xshift=1.5mm]de.south) {problem-level reading:\\10 queries, blind};""",
    r"""\node[box=30mm, anchor=north] (cat) at ([yshift=-5mm]de.south) {category proxy: all test queries,\\paired tests, dev-set sensitivity};""")
rep(r"""\draw[arr] (j) -| (cat.north);
\draw[arr] (j) -| (man.north);""", r"""\draw[arr] (j) -| (cat.north);""")
rep(r"""The problem-level reading below measures both failure modes directly.""",
    r"""Section~VII discusses what this means for interpreting the results.""")
a = s.index(r"\textit{Problem-level reading.}")
b = s.index(r"\section{Results}")
s = s[:a] + s[b:]
a = s.index(r"\subsection{Problem-level reading}")
b = s.index(r"\section{Discussion}")
s = s[:a] + s[b:]
rep(r"""The problem-level reading qualifies the main tables. Category agreement counts""",
    r"""The proxy itself qualifies the main tables. Category agreement counts""")
rep(r"""\textit{Limitations.} The relevance label is a proxy, and the problem-level check covers 10 queries and one annotator, so it estimates the direction and rough size of the disagreement rather than precise reader precision.""",
    r"""\textit{Limitations.} The relevance label is a proxy, and we did not collect problem-level judgments, so the results rank systems but do not measure how many retrieved papers a researcher would keep.""")
rep(r"""A blind problem-level reading shows that category agreement and same-problem relevance often disagree, which argues for reporting category precision together with at least a small problem-level judgment. Future work will add expert judgments from several annotators and full-text and citation signals.""",
    r"""Because category agreement is a proxy, the next step is a problem-level test collection judged by several annotators on the released rankings, together with full-text and citation signals.""")
rep(r"""per-query scores, rankings and the manual labels are available""", r"""per-query scores and rankings are available""")
rep(r"\val{ai.disclosure}", r"Generative AI (Claude, Anthropic) was used to help write the analysis scripts and edit the manuscript text. The authors designed the study, ran all experiments, verified all code, results and references, and take full responsibility for the content.")
dst.write_text(s, encoding="utf-8")
print("written", dst)
