"""Bandingkan label notebook dengan label_folds_v6.csv dan tulis draft ViT serta CNN. Tidak membaca temuan.

    python temp/nb_final/cek_hasil.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
a = pd.read_csv(OUT / "label_final.csv", dtype={"image_id": str}).set_index("image_id")
b = pd.read_csv(ROOT / "temp" / "eda_114_paper" / "output" / "label_folds_v6.csv", dtype={"image_id": str}).set_index("image_id")
sama = a.index.intersection(b.index)
print("notebook", len(a), "| skrip v6", len(b), "| irisan", len(sama), "| hanya notebook", len(a.index.difference(b.index)), "| hanya skrip", len(b.index.difference(a.index)))
beda = sama[a.loc[sama, "label"].values != b.loc[sama, "label"].values]
print("label beda di irisan:", len(beda), f"({len(beda) / len(sama):.4f})")
if len(beda):
    print(pd.crosstab(b.loc[beda, "label"].rename("skrip"), a.loc[beda, "label"].rename("notebook")).to_string())
print("per kelas notebook:", a.label.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())
print("per kelas skrip   :", b.label.value_counts().reindex(NAMES).fillna(0).astype(int).to_dict())

C = OUT / "cache"
ids = np.load(ROOT / "temp" / "exp_006_cluster_dinov3" / "output" / "ids_test.npy").astype(str)
pv, pc = np.load(C / "vit_v6_test.npy"), np.load(C / "cnn_v6_test.npy")
for n, p in (("vit", pv), ("cnn", pc)):
    pd.DataFrame({"image_id": ids, "waste_category": np.array(NAMES)[p.argmax(1)]}).to_csv(OUT / f"draft_{n}.csv", index=False)
print("setuju ViT lawan CNN:", round(float((pv.argmax(1) == pc.argmax(1)).mean()), 4), "beda", int((pv.argmax(1) != pc.argmax(1)).sum()))
print(pd.read_csv(OUT / "submission.csv", dtype=str).waste_category.value_counts().reindex(NAMES).to_dict())
