#!/usr/bin/env python3
"""Step 2: encode a corpus with the extra dense models used in the revision.

Usage (from the article1_ictacs2026 folder):
    python revision/scripts/02_encode.py --corpus may2026
    python revision/scripts/02_encode.py --corpus y2020_2024

Jobs per corpus (each saved as its own .npy, so an interrupted run resumes):
  may2026     minilm_ta (reproducibility check of the stored matrix),
              minilm_title, minilm_abstract (ablation), specter2_ta, bge_small_ta
  y2020_2024  minilm_ta, specter2_ta, bge_small_ta

Text fields follow the original pipeline: title + " " + abstract for
sentence-transformers models; title + [SEP] + abstract for SPECTER2/SciNCL,
as in their model cards. All vectors are L2-normalized float32.
If the `adapters` package is unavailable, SPECTER2 is replaced by SciNCL
(malteos/scincl) and the metadata file says so.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # .../article1_ictacs2026/revision
PAPER_DIR = ROOT.parents[1]                         # .../Paper
EMB = ROOT / "embeddings"
LOGS = ROOT / "logs"
CORPORA = {
    # repository layout first, then the original project layout
    "may2026": (ROOT / "data" / "arxiv_may2026.jsonl") if (ROOT / "data" / "arxiv_may2026.jsonl").exists()
               else PAPER_DIR / "experiments" / "data" / "raw" / "arxiv_papers.jsonl",
    "y2020_2024": ROOT / "data" / "arxiv_2020_2024.jsonl",
}
JOBS = {
    "may2026": ["minilm_ta", "minilm_title", "minilm_abstract", "specter2_ta", "bge_small_ta"],
    "y2020_2024": ["minilm_ta", "specter2_ta", "bge_small_ta"],
}
SEP_MARK = "\u241e"  # placeholder, replaced by tokenizer.sep_token
ST_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge_small": "BAAI/bge-small-en-v1.5",
}


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / "02_encode.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_corpus(name: str) -> list[dict]:
    path = CORPORA[name]
    if not path.exists():
        sys.exit(f"Corpus file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def model_sha(model_id: str) -> str:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def encode_st(model_id: str, texts: list[str], device: str) -> tuple[np.ndarray, dict]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)
    vec = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    return vec.astype(np.float32), {"max_seq_length": int(model.max_seq_length)}


def encode_cls(texts: list[str], device: str) -> tuple[np.ndarray, dict]:
    """SPECTER2 (proximity adapter) if `adapters` is installed, otherwise SciNCL."""
    import torch
    from transformers import AutoTokenizer

    try:
        from adapters import AutoAdapterModel

        tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        used = {"model": "allenai/specter2_base + adapter allenai/specter2 (proximity)",
                "sha_base": model_sha("allenai/specter2_base"), "sha_adapter": model_sha("allenai/specter2")}
    except Exception as exc:  # noqa: BLE001
        log(f"SPECTER2 unavailable ({type(exc).__name__}: {exc}); falling back to SciNCL")
        from transformers import AutoModel

        tok = AutoTokenizer.from_pretrained("malteos/scincl")
        model = AutoModel.from_pretrained("malteos/scincl")
        used = {"model": "malteos/scincl", "sha": model_sha("malteos/scincl"), "fallback_reason": str(exc)}

    model.to(device).eval()
    sep = tok.sep_token
    # texts arrive as title + SEP_MARK + abstract; the marker becomes the tokenizer's sep token
    full = [t.replace(SEP_MARK, sep) for t in texts]
    order = np.argsort([len(t) for t in full])
    out = np.zeros((len(full), model.config.hidden_size), dtype=np.float32)
    bs = 16
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            batch = tok([full[j] for j in idx], padding=True, truncation=True, max_length=512,
                        return_tensors="pt", return_token_type_ids=False).to(device)
            h = model(**batch).last_hidden_state[:, 0, :]
            h = torch.nn.functional.normalize(h, dim=-1)
            out[idx] = h.cpu().numpy()
            done = i + len(idx)
            if (i // bs) % 50 == 0 or done == len(order):
                el = time.perf_counter() - t0
                log(f"    {done}/{len(order)} docs, {done/el:.1f} docs/s, ETA {(len(order)-done)/(done/el)/60:.1f} min")
    used["max_length"] = 512
    used["pooling"] = "CLS"
    return out, used


def texts_for(job: str, rows: list[dict]) -> list[str]:
    field = job.rsplit("_", 1)[1]
    if job.startswith(("specter2", "scincl")):
        return [r["title"] + SEP_MARK + r["abstract"] for r in rows]
    if field == "ta":
        return [r["title"] + " " + r["abstract"] for r in rows]
    if field == "title":
        return [r["title"] for r in rows]
    if field == "abstract":
        return [r["abstract"] for r in rows]
    raise ValueError(job)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    p.add_argument("--only", nargs="*", help="run only these jobs")
    args = p.parse_args()

    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_corpus(args.corpus)
    EMB.mkdir(parents=True, exist_ok=True)
    ids_path = EMB / f"{args.corpus}__ids.txt"
    ids_path.write_text("\n".join(r["arxiv_id"] for r in rows) + "\n", encoding="utf-8")
    log(f"corpus {args.corpus}: {len(rows)} records, device={device}, threads={torch.get_num_threads()}")

    import transformers

    versions = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
                "transformers": transformers.__version__, "numpy": np.__version__}
    try:
        import sentence_transformers

        versions["sentence_transformers"] = sentence_transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import adapters

        versions["adapters"] = adapters.__version__
    except Exception:  # noqa: BLE001
        versions["adapters"] = "not installed"

    for job in args.only or JOBS[args.corpus]:
        out_npy = EMB / f"{args.corpus}__{job}.npy"
        out_meta = EMB / f"{args.corpus}__{job}.json"
        if out_npy.exists() and out_meta.exists():
            log(f"skip {job}: already done")
            continue
        log(f"start {job}")
        texts = texts_for(job, rows)
        t0 = time.perf_counter()
        if job.startswith("specter2"):
            vec, info = encode_cls(texts, device)
        else:
            mid = ST_MODELS[job.rsplit("_", 1)[0]]
            vec, info = encode_st(mid, texts, device)
            info.update({"model": mid, "sha": model_sha(mid)})
        secs = time.perf_counter() - t0
        np.save(out_npy, vec)
        meta = {"corpus": args.corpus, "job": job, "n": int(vec.shape[0]), "dim": int(vec.shape[1]),
                "seconds": round(secs, 1), "docs_per_s": round(len(texts) / secs, 2), "device": device,
                "normalized": True, "row_order": ids_path.name, **info, "versions": versions,
                "finished_at": datetime.now(timezone.utc).isoformat()}
        out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"done {job}: shape {vec.shape}, {secs/60:.1f} min")
    log("all jobs finished")


if __name__ == "__main__":
    main()
