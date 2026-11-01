"""Validasi format kandidat test dan berkas label. Tidak membaca temuan.

    python temp/eda_110_fusi/cek_kandidat.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
NAMES = {"glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"}
s = pd.read_csv(ROOT / "data" / "SampleSubmission_DSC.csv", dtype=str)
lama = pd.read_csv(ROOT / "temp" / "exp_006_cluster_dinov3" / "output" / "draft_submission_dinov3_cluster.csv", dtype=str).set_index("image_id")
for v in ("t1_in1k", "t1_in1k_coco_abst", "t05_abst"):
    for j in ("lr", "vote"):
        f = OUT / f"kandidat_test_{v}_{j}.csv"
        d = pd.read_csv(f, dtype=str)
        ok = list(d.columns) == list(s.columns) and len(d) == 480 and set(d.image_id) == set(s.image_id) and set(d.waste_category) <= NAMES
        ubah = d.set_index("image_id").loc[lama.index, "waste_category"].ne(lama["waste_category"])
        pl = (lama["waste_category"] == "plastic") & ubah
        print(f.name, "valid" if ok else "TIDAK VALID", "beda dari EXP-006:", int(ubah.sum()), "| dari plastic:", int(pl.sum()))
    l = pd.read_csv(OUT / f"label_folds_{v}.csv", dtype={"image_id": str})
    print("  label", len(l), "hold-out", int((l.fold == 0).sum()), "group tersentuh dua fold:", int((l.groupby("group").fold.nunique() > 1).sum()))
