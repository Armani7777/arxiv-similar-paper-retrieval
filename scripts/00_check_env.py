#!/usr/bin/env python3
"""Step 0: check that this computer can run the revision scripts.

Prints Python/package versions, CPU/GPU, free disk space, and whether
export.arxiv.org and huggingface.co are reachable. Nothing is downloaded
except two tiny HTTP responses.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "00_check_env.json"


def check_import(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "ok")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({type(exc).__name__}: {exc})"


def check_url(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "paper703-revision-check"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return f"OK (HTTP {r.status})"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL ({type(exc).__name__}: {exc})"


def main() -> None:
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "free_disk_gb_here": round(shutil.disk_usage(ROOT).free / 1e9, 1),
        "free_disk_gb_home": round(shutil.disk_usage(Path.home()).free / 1e9, 1),
    }
    for pkg in ["numpy", "tqdm", "arxiv", "torch", "transformers", "sentence_transformers", "adapters", "huggingface_hub"]:
        info[pkg] = check_import(pkg)
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        info["cuda_available"] = False
    info["arxiv_api"] = check_url("https://export.arxiv.org/api/query?search_query=cat:cs.IR&max_results=1")
    info["huggingface"] = check_url("https://huggingface.co/api/models/allenai/specter2_base")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in info.items():
        print(f"{k:24s} {v}")

    problems = [k for k, v in info.items() if isinstance(v, str) and (v.startswith("MISSING") or v.startswith("FAIL"))]
    py_ok = (3, 10) <= sys.version_info[:2] <= (3, 13)
    print()
    if not py_ok:
        print("WARNING: Python 3.10-3.13 is expected.")
    if info["free_disk_gb_home"] < 5:
        print("WARNING: less than 5 GB free; models and torch need about 3-4 GB.")
    if problems:
        print("PROBLEMS:", ", ".join(problems))
        if problems == ["adapters"]:
            print("Only 'adapters' is missing: scripts will fall back to SciNCL instead of SPECTER2.")
    else:
        print("ALL OK")
    print(f"Saved to {LOG}")


if __name__ == "__main__":
    main()
