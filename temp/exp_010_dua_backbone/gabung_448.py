"""Draft CNN 448 dan gabungannya dengan ViT 384. Tidak membaca temuan.

    python temp/exp_010_dua_backbone/gabung_448.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
IDS = np.load(ROOT / "temp" / "exp_006_cluster_dinov3" / "output" / "ids_test.npy").astype(str)

vit = np.load(OUT / "vit_v5" / "test_fold0.npy")
cnn = np.load(OUT / "cnn_v5_448" / "test_fold0.npy")
cnn384 = np.load(OUT / "cnn_v5" / "test_fold0.npy")
for nama, p in (("cnn448", cnn), ("gabungan448", (vit + cnn) / 2)):
    pred = np.array(NAMES)[p.argmax(1)]
    pd.DataFrame({"image_id": IDS, "waste_category": pred}).to_csv(OUT / f"draft_{nama}_v5.csv", index=False)
    print(nama, dict(zip(NAMES, np.bincount(p.argmax(1), minlength=8).tolist())))
print("setuju cnn 384 lawan 448:", round(float((cnn.argmax(1) == cnn384.argmax(1)).mean()), 4), "| beda", int((cnn.argmax(1) != cnn384.argmax(1)).sum()))
print("setuju vit lawan cnn448:", round(float((cnn.argmax(1) == vit.argmax(1)).mean()), 4))
