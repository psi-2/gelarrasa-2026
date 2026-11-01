"""Isi keluarga hash besar dan konsistensi label otomatis pada near-duplicate. Perhitungan ringan, tanpa membuka citra.

    python temp/eda_104_duplikat/konsistensi.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
F = ROOT / "temp" / "eda_110_fusi" / "output"
NAMES = ["glass", "paper", "electronic", "metal", "plastic", "cardboard", "textile", "foodwaste"]
pd.set_option("display.width", 200, "display.max_columns", 30)
k = pd.read_csv(OUT / "keluarga.csv", dtype={"image_id": str})
m = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
g = pd.read_csv(ROOT / "temp" / "eda_107_visual" / "output" / "gabung.csv", dtype={"image_id": str})
d = k.merge(m[["split", "image_id", "fam", "wh", "nama"]], on=["split", "image_id"]).merge(g[["split", "image_id", "lok_halus", "kontras"]], on=["split", "image_id"])
for n in sorted(d.kel_n.unique())[-2:]:
    s = d[d.kel_n == n]
    print(f"keluarga n={n}: fam {s.fam.value_counts().to_dict()}, split {s.split.value_counts().to_dict()}, nama {s.nama.value_counts().to_dict()}, lok_halus median {s.lok_halus.median():.3f}, kontras median {s.kontras.median():.3f}")

tr = m[m.split == "train"].sort_values("image_id")
te = m[m.split == "test"].sort_values("image_id")
base = pd.concat([tr, te], ignore_index=True)[["split", "image_id"]]
p = pd.read_csv(OUT / "pasangan_hash.csv", dtype={"ia": str, "ib": str})
for var in ("t1_in1k", "t1_in1k_coco_abst", "t05_abst"):
    V = np.load(F / f"var_{var}.npy")
    b = base.assign(y=V.argmax(1))
    lk = {(s, i): y for s, i, y in b.itertuples(index=False)}
    ya = np.array([lk[(s, i)] for s, i in zip(p.sa, p.ia)])
    yb = np.array([lk[(s, i)] for s, i in zip(p.sb, p.ib)])
    print(var, "pasangan Hamming <=4:", round(float((ya == yb)[p.ham <= 4].mean()), 4), "| <=10:", round(float((ya == yb).mean()), 4),
          "| silang train-test <=4:", round(float((ya == yb)[(p.ham <= 4) & (p.sa != p.sb)].mean()), 4))
