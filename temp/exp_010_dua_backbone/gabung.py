"""Gabungkan prediksi ViT dan CNN lewat rata-rata probabilitas. Tidak membaca temuan.

    python temp/exp_010_dua_backbone/gabung.py --versi v5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
IDS = np.load(ROOT / "temp" / "exp_006_cluster_dinov3" / "output" / "ids_test.npy").astype(str)
SS = pd.read_csv(ROOT / "data" / "SampleSubmission_DSC.csv", dtype=str)

ap = argparse.ArgumentParser()
ap.add_argument("--versi", default="v5")
v = ap.parse_args().versi

vit = np.load(OUT / f"vit_{v}" / "test_fold0.npy")
cnn = np.load(OUT / f"cnn_{v}" / "test_fold0.npy")
gab = (vit + cnn) / 2
pv, pc, pg = [np.array(NAMES)[p.argmax(1)] for p in (vit, cnn, gab)]

for nama, pred in (("vit", pv), ("cnn", pc), ("gabungan", pg)):
    sub = pd.DataFrame({"image_id": IDS, "waste_category": pred})
    sub.to_csv(OUT / f"draft_{nama}_{v}.csv", index=False)
    assert (sub["image_id"].values == SS["image_id"].values).all() and sub["waste_category"].isin(NAMES).all()

hasil = {
    "setuju_vit_cnn_test": round(float((pv == pc).mean()), 4),
    "beda_vit_cnn_test": int((pv != pc).sum()),
    "distribusi_gabungan": pd.Series(pg).value_counts().reindex(NAMES).fillna(0).astype(int).to_dict(),
    "prob_maks_rata": round(float(gab.max(1).mean()), 4),
    "porsi_prob_maks_lt_0.7": round(float((gab.max(1) < 0.7).mean()), 4),
}
print(json.dumps(hasil, indent=2, ensure_ascii=False))
(OUT / f"gabung_{v}.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
