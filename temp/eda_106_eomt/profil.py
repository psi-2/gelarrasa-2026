"""Profil kelas EoMT per cluster K=64 dan per keluarga sumber. Hanya kelas objek, kelas latar dicetak terpisah.

    python temp/eda_106_eomt/profil.py coco
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 300, "display.max_colwidth", 110)
mdl = sys.argv[1] if len(sys.argv) > 1 else "coco"
kel = json.loads((OUT / f"{mdl}_kelas.json").read_text(encoding="utf-8"))
g = pd.read_csv(ROOT / "temp" / "eda_102_sumber" / "output" / "meta_assign.csv", dtype={"image_id": str})
tr = g[g.split == "train"].sort_values("image_id")
te = g[g.split == "test"].sort_values("image_id")
d = pd.concat([tr, te], ignore_index=True)
A = np.vstack([np.load(OUT / f"{mdl}_train.npy"), np.load(OUT / f"{mdl}_test.npy")])
C = A[:, :, 1]  # area tengah
LATAR = {"coco": ["table-merged", "wall-other-merged", "floor-other-merged", "floor-wood", "wall-wood", "wall-tile", "wall-stone", "wall-brick", "dirt-merged",
                  "rock-merged", "dining table", "counter", "mirror-stuff", "ceiling-merged", "cabinet-merged", "shelf", "person", "pavement-merged", "grass-merged",
                  "sand", "snow", "sky-other-merged", "water-other", "window-other", "door-stuff", "bed", "gravel", "road", "platform"],
         "ade": ["wall", "floor", "table", "ceiling", "earth", "person", "desk", "cabinet", "sidewalk", "road", "grass", "sand", "windowpane", "door", "countertop", "shelf", "water", "rock", "building", "sky"]}[mdl]
ob = np.array([k not in LATAR for k in kel])
print("median area latar total", round(float(np.median(C[:, ~ob].sum(1))), 3))
rows = []
nam = np.array(kel)
for c in sorted(d.cluster.unique()):
    m = (d.cluster == c).values
    r = C[m].mean(0).copy()
    r[~ob] = 0
    u = r.argsort()[::-1][:6]
    rows.append({"c": c, "nama": d.nama[m].iloc[0], "porsi": d.porsi[m].iloc[0], "lolos": d.lolos[m].iloc[0], "n": int(m.sum()), "te": int((m & (d.split == "test").values).sum()),
                 "fam": d.fam[m].value_counts().index[0], "top": ", ".join(f"{nam[i]} {r[i]:.2f}" for i in u if r[i] >= .01)})
t = pd.DataFrame(rows).sort_values(["nama", "te"], ascending=[True, False])
print(t.to_string(index=False))
t.to_csv(OUT / f"{mdl}_profil_cluster.csv", index=False)
print("\nkelas objek dengan area rata-rata tengah terbesar (semua citra)")
r = C.mean(0).copy()
r[~ob] = 0
print(", ".join(f"{nam[i]} {r[i]:.3f}" for i in r.argsort()[::-1][:30]))
print("\nper keluarga dan split")
for (f, s), gg in d.groupby(["fam", "split"]):
    if len(gg) < 15:
        continue
    r = C[gg.index].mean(0).copy()
    r[~ob] = 0
    print(f"{f:14s} {s:5s} {len(gg):4d} " + ", ".join(f"{nam[i]} {r[i]:.2f}" for i in r.argsort()[::-1][:7]))
