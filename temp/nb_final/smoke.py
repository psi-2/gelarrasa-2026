"""Smoke test notebook: baris magic dibuang, lalu tiap code cell dijalankan berurutan dengan CFG.SMOKE_TEST=True.

smoke_notebook.py memakai exec biasa, sehingga baris `!` dan `%` tidak bisa dikompilasi. Selain itu sama.

    python temp/nb_final/smoke.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
nb = json.loads((HERE / "dsc_gelar_rasa_2026_dinov3_convnext_normal.ipynb").read_text(encoding="utf-8"))
ns = {}
t0 = time.time()
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(l for l in "".join(c["source"]).splitlines(keepends=True) if not l.lstrip().startswith(("!", "%pip")))
    if not src.strip():
        continue
    print(f"RUN cell {i} [{time.time() - t0:.0f}s]: {src.strip().splitlines()[0][:80]}", flush=True)
    exec(compile(src, f"cell-{i}", "exec"), ns)
    if "settings" in c["metadata"].get("tags", []):
        cfg = ns["CFG"]
        cfg.SMOKE_TEST = True
        cfg.OUTPUT_DIR = HERE / "smoke_out"
        cfg.EPOCHS = 1
print("SMOKE TEST LULUS", round(time.time() - t0), "detik")
